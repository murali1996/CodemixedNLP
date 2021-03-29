import datetime
import json
import logging
import os
import sys
import time
from copy import deepcopy
from typing import List, Union

import numpy as np
import torch
from pytorch_pretrained_bert import BertAdam
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer

from .helpers import Tokenizer
from .helpers import batch_iter, progress_bar
from .helpers import create_vocab, load_vocab
from .models_unified import Model
from ..datasets import read_datasets_jsonl_new, create_path, EXAMPLE

LOGS_FOLDER = "./logs"
printlog = print


class DatasetArgs:
    def __init__(self,
                 dataset_folder: str = None,
                 augment_train_datasets: List[str] = None,
                 **kwargs):
        super(DatasetArgs, self).__init__(**kwargs)

        self.dataset_folder = dataset_folder
        self.augment_train_datasets = augment_train_datasets

        self.validate_dataset_folder()

    def validate_dataset_folder(self):
        assert os.path.exists(self.dataset_folder)


class ModelArgs:
    def __init__(self,
                 encoders: Union[str, List[str]],
                 encoder_attributes: Union[dict, List[dict]],
                 task_attributes: Union[dict, List[dict]],
                 target_label_fields: Union[str, List[str]] = None,  # must be of same length as `task_attributes`
                 max_len: int = 200,
                 encodings_merge_type: str = "concat",
                 **kwargs):
        super(ModelArgs, self).__init__(**kwargs)

        if isinstance(encoders, str):
            assert isinstance(encoder_attributes, dict)
            encoders = [encoders, ]
            encoder_attributes = [encoder_attributes, ]
        elif isinstance(encoders, list):
            assert isinstance(encoder_attributes, list)
            assert len(encoders) == len(encoder_attributes)

        if isinstance(task_attributes, dict):
            assert isinstance(target_label_fields, str)
            task_attributes = [task_attributes, ]
            target_label_fields = [target_label_fields, ]
        elif isinstance(task_attributes, list):
            assert isinstance(target_label_fields, list)
            assert len(task_attributes) == len(target_label_fields)

        self.encoders = [enc.lower() for enc in encoders]
        self.encoder_attributes = encoder_attributes
        self.task_attributes = task_attributes
        self.target_label_fields = target_label_fields
        self.max_len = max_len
        self.encodings_merge_type = encodings_merge_type


class TraintestArgs:
    def __init__(self,
                 mode: Union[List[str], str] = None,
                 text_field: str = None,
                 text_aug_fields: List[str] = None,
                 tag_input_label_field: str = None,  # field in dataset for tags to augment text field (eg. "lang_ids")
                 fusion_text_fields: List = None,  # overwrites `text_field`
                 fusion_strategy: str = None,
                 max_epochs: int = 5,
                 patience: int = 4,
                 batch_size: int = 16,
                 grad_acc=2,
                 checkpoint_using_accuracy: bool = False,
                 save_errors_path: str = None,
                 eval_ckpt_path: str = None,
                 debug: bool = False,
                 **kwargs):
        super(TraintestArgs, self).__init__(**kwargs)

        if isinstance(mode, str):
            mode = [mode, ]
        self.mode = mode
        # inputs
        #   this ...
        self.text_field = text_field
        self.text_aug_fields = text_aug_fields
        self.tag_input_label_field = tag_input_label_field
        #   or ...
        if fusion_text_fields:
            if self.text_field or self.text_aug_fields or self.tag_input_label_field:
                raise ValueError("cannot specify both `text_field` as well as `fusion_text_fields`")
            assert len(fusion_text_fields) > 1
            if not fusion_strategy:
                fusion_strategy = fusion_strategy or "concat"
                print(f"`fusion_strategy` set to default `concat`; other available choices: [`meanpool`, `maxpool`]\n")
        else:
            self.text_field = self.text_field or "text"  # set it to a default if None is given
        self.fusion_text_fields = fusion_text_fields
        self.n_fields_fusion = len(self.fusion_text_fields) if self.fusion_text_fields else None
        self.fusion_strategy = fusion_strategy
        # others
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.grad_acc = grad_acc
        self.checkpoint_using_accuracy = checkpoint_using_accuracy
        self.save_errors_path = save_errors_path
        self.debug = debug
        # evaluation
        self.eval_ckpt_path = eval_ckpt_path

        self.validate_mode()

    def validate_mode(self):
        assert all([m in ["train", "test", "interactive"] for m in self.mode])


class Args(DatasetArgs, ModelArgs, TraintestArgs):

    def __init__(self, **kwargs):
        super(Args, self).__init__(**kwargs)
        self.validate_args()

    def validate_args(args):

        if args.batch_size:
            assert args.batch_size > 0, printlog(f"batch_size ({args.batch_size}) must be a positive integer")

        if args.tag_input_label_field:
            if not (len(args.encoders) == 1 and args.encoders[0] == "bert"):
                raise NotImplementedError("for `tag_input_label_field`, specify just bert encoder and a single task")

        return


def _get_checkpoints_path(args):
    if args.eval_ckpt_path:
        assert all([ii in ["test", "interactive"] for ii in args.mode])
        ckpt_path = args.eval_ckpt_path
    else:
        assert args.mode[0] == "train", print("`mode` must first contain `train` if no eval ckpt path is specified")
        ckpt_path = os.path.join(args.dataset_folder,
                                 "checkpoints",
                                 f'{str(datetime.datetime.now()).replace(" ", "_")}')
        if os.path.exists(ckpt_path):
            msg = f"ckpt_path: {ckpt_path} already exists. Did you mean to set mode to `test` ?"
            raise Exception(msg)
        create_path(ckpt_path)
    printlog(f"ckpt_path: {ckpt_path}")
    return ckpt_path


def _set_logger(args, ckpt_path):
    global printlog

    if args.debug:
        logger_file_name = None
        printlog = print
    else:
        if not os.path.exists(os.path.join(ckpt_path, LOGS_FOLDER)):
            os.makedirs(os.path.join(ckpt_path,
                                     LOGS_FOLDER))
        # logger_file_name = os.path.join(ckpt_path, LOGS_FOLDER, "{}_{}".format(
        #     os.path.basename(__file__).split('.')[-2], args.model_name))
        logger_file_name = os.path.join(ckpt_path,
                                        LOGS_FOLDER,
                                        f'{str(datetime.datetime.now()).replace(" ", "_")}')
        logging.basicConfig(level=logging.INFO, filename=logger_file_name, filemode='a',
                            datefmt='%Y-%m-%d:%H:%M:%S',
                            format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s')
        printlog = logging.info

    print(f"logger_file_name: {logger_file_name}")

    printlog('\n\n\n--------------------------------\nBeginning to log...\n--------------------------------\n')
    printlog(" ".join(sys.argv))
    return logger_file_name


def _load_train_examples(args):
    train_examples = read_datasets_jsonl_new(os.path.join(args.dataset_folder, f"train.jsonl"), "train")
    if args.text_field and args.text_aug_fields:
        new_train_examples = []
        for ex in train_examples:
            for field in args.text_aug_fields:
                new_example = deepcopy(ex)
                new_example.uid = ex.uid + f"_{field}"
                new_example[args.text_field] = new_example[field]
                new_train_examples.append(new_example)
        train_examples.extend(new_train_examples)
        printlog(f"train examples increased to {len(train_examples)} due to `text_aug_fields`")
    if args.augment_train_datasets:
        train_augment_examples = []
        for name in args.augment_train_datasets:
            temp_examples = read_datasets_jsonl_new(os.path.join(name, "train.jsonl"), "train")
            train_augment_examples.extend(temp_examples)
        train_examples.extend(train_augment_examples)
        printlog(f"train examples increased to {len(train_examples)} due to `augment_train_datasets`")
    return train_examples


def run_unified(args: Args = None, **kwargs):
    args = args or Args(**kwargs)

    """ basics """
    ckpt_path = _get_checkpoints_path(args)
    logger_file_name = None
    if not args.debug:
        logger_file_name = _set_logger(args, ckpt_path)

    """ settings """
    start_epoch, n_epochs = 0, args.max_epochs
    train_batch_size, dev_batch_size = (args.batch_size, args.batch_size)
    if args.fusion_text_fields:
        train_batch_size = int(train_batch_size / args.n_fields_fusion)
        dev_batch_size = int(dev_batch_size / args.n_fields_fusion)
        args.grad_acc *= args.n_fields_fusion

    """ load dataset """
    train_examples, dev_examples, test_examples = [], [], []
    args.mode = args.mode  # already a list
    assert len(args.mode), printlog("Expected at least one mode amng `train`, `test`, `interactive`")
    for mode in args.mode:
        if "train" in mode:
            train_examples = _load_train_examples(args)
            dev_examples = read_datasets_jsonl_new(os.path.join(args.dataset_folder, f"dev.jsonl"), f"dev")
        elif "test" in mode:  # can be anything like "test", "test-xyz", "test-collate", etc.
            test_examples = read_datasets_jsonl_new(os.path.join(args.dataset_folder, f"test.jsonl"), f"test")
        elif "interactive" in mode:
            continue
        else:
            raise ValueError(f"invalid mode `{mode}` encountered in {args.mode}")
    if args.debug:
        printlog("debug mode enabled; reducing dataset size to 40 (train) and 20 (dev/test) resp.")
        train_examples = train_examples[:40]
        dev_examples = dev_examples[:20]
        test_examples = test_examples[:20]

    """ load tokenizers """
    tokenizers = []
    for iii, (enc_name, enc_attr) in enumerate(zip(args.encoders, args.encoder_attributes)):
        tokenizer_ckpt_path = os.path.join(ckpt_path, f"encoder_{iii}")
        create_path(tokenizer_ckpt_path)
        if enc_name == "bert":
            tokenizer = Tokenizer(bert_tokenizer=AutoTokenizer.from_pretrained(enc_attr["pretrained_name_or_path"]))
            if args.tag_input_label_field:
                if any(["train" in m for m in args.mode]):
                    data = [i for ex in train_examples for i in getattr(ex, args.tag_input_label_field).split(" ")]
                    tokenizer.load_tag_vocab(data)
                    tokenizer.save_tag_vocab_to_checkpoint(tokenizer_ckpt_path)
                else:
                    tokenizer.load_tag_vocab_from_checkpoint(tokenizer_ckpt_path)
        elif enc_name == "lstm":
            tokenizer = Tokenizer()
            if enc_attr["input_representation"] in ["char", "sc"]:
                if any(["train" in m for m in args.mode]):
                    data = [getattr(ex, field) for ex in train_examples for field in
                            args.fusion_text_fields] if args.fusion_text_fields else [getattr(ex, args.text_field) for
                                                                                      ex in train_examples]
                    tokenizer.load_word_vocab(data)
                    tokenizer.save_word_vocab_to_checkpoint(tokenizer_ckpt_path)
                else:
                    tokenizer.load_word_vocab_from_checkpoint(tokenizer_ckpt_path)
        else:
            raise ValueError
        tokenizers.append(tokenizer)

    """ load targets """
    target_label_vocabs = []
    for iii, (target_label_field, task_attribute) in enumerate(zip(args.target_label_fields, args.task_attributes)):
        target_label_ckpt_path = os.path.join(ckpt_path, f"task_{iii}")
        create_path(target_label_ckpt_path)
        target_label_ckpt_path = os.path.join(target_label_ckpt_path, "target_label_vocab.json")
        if any(["train" in m for m in args.mode]):
            if not getattr(train_examples[0], target_label_field):
                raise ValueError(f"Unable to find {target_label_field} target_label_field in the training file")
            data = [getattr(ex, target_label_field) for ex in train_examples]
            if any([True if d is None or d == "" else False for d in data]):
                raise ValueError(f"Found empty strings when obtaining {target_label_field} target_label_field")
            if task_attribute["name"] == "classification":
                vcb = create_vocab(data, is_label=True, labels_data_split_at_whitespace=False)
            elif task_attribute["name"] == "seq_tagging":
                vcb = create_vocab(data, is_label=True, labels_data_split_at_whitespace=True)
            else:
                raise ValueError
            json.dump(vcb._asdict(), open(target_label_ckpt_path, "w"), indent=4)
        else:
            vcb = load_vocab(target_label_ckpt_path)
        target_label_vocabs.append(vcb)

    """ define model and optimizers """
    model, optimizers = None, []
    # populate encoder attributes
    for i, (tokenizer, enc_name, enc_attr) in enumerate(zip(tokenizers, args.encoders, args.encoder_attributes)):
        if enc_name == "bert":
            if args.tag_input_label_field:
                enc_attr.update({"n_token_type_ids": tokenizer.tag_input_label_vocab.n_all_tokens})
                args.encoder_attributes[i] = enc_attr
        elif enc_name == "lstm":
            if enc_attr["input_representation"] == "char":
                enc_attr.update({
                    "embdim": 128,
                    "hidden_size": 128,
                    "num_layers": 1,
                    "n_embs": len(tokenizer.word_vocab.chartoken2idx),
                    "padding_idx": tokenizer.word_vocab.char_pad_token_idx
                })
                args.encoder_attributes[i] = enc_attr
            elif enc_attr["input_representation"] == "sc":
                enc_attr.update({
                    "embdim": 3 * len(tokenizer.word_vocab.chartoken2idx),
                    "hidden_size": 256,
                    "num_layers": 2
                })
                args.encoder_attributes[i] = enc_attr
            elif enc_attr["input_representation"] == "fasttext":
                enc_attr.update({
                    "embdim": 300,
                    "hidden_size": 256,
                    "num_layers": 2
                })
                args.encoder_attributes[i] = enc_attr
    # populate task attributes
    for i, (target_label_vocab, task_attr) in enumerate(zip(target_label_vocabs, args.task_attributes)):
        task_attr.update({
            "nlabels": target_label_vocab.n_all_tokens
        })
        args.task_attributes[i] = task_attr
    # load model
    model = Model(
        encoders=args.encoders,
        encoder_attributes=args.encoder_attributes,
        task_attributes=args.task_attributes,
        n_fields_fusion=args.n_fields_fusion,
        fusion_strategy=args.fusion_strategy,
        encodings_merge_type=args.encodings_merge_type
    )
    # load optimizers
    model_resources = model.all_resources()
    for resc in model_resources:
        if resc.requires_bert_optimizer:
            params = [param for param in list(resc.named_parameters())]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            t_total = int(len(train_examples) / train_batch_size / args.grad_acc * n_epochs)
            lr = 2e-5  # 1e-4 or 2e-5 or 5e-5
            optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.1, t_total=t_total)
            printlog(f"{len(params)} number of params are being optimized with BertAdam")
        else:
            params = [param for param in list(resc.parameters())]
            if len(params) == 0:
                optimizer = None
                print(f"Warning: optimizer got an empty parameter list in `{resc.name}`")
            else:
                optimizer = torch.optim.Adam(resc.parameters(), lr=0.001)
                printlog(f"{len(params)} number of params are being optimized with Adam")
        optimizers.append(optimizer)
    printlog(f"number of parameters (all, trainable) in your model: {model.get_model_nparams()}")

    """ training and validation """
    if "train" in args.mode:
        best_dev_loss, best_dev_loss_epoch = 0., -1
        best_dev_acc, best_dev_acc_epoch = 0., -1
        best_dev_f1, best_dev_f1_epoch = 0., -1
        for epoch_id in range(start_epoch, n_epochs):

            if epoch_id - best_dev_acc_epoch > args.patience and epoch_id - best_dev_f1_epoch > args.patience:
                printlog(f"set patience of {args.patience} epochs reached; terminating train process")
                break

            printlog("\n\n################")
            printlog(f"epoch: {epoch_id}")

            """ training """
            train_loss, train_acc, train_preds = 0., -1, []
            n_batches = int(np.ceil(len(train_examples) / train_batch_size))
            train_examples_batch_iter = batch_iter(train_examples, train_batch_size, shuffle=True)
            printlog(f"len of train data: {len(train_examples)}")
            printlog(f"n_batches of train data: {n_batches}")
            for optimizer in optimizers:
                if optimizer:
                    optimizer.zero_grad()
            model.train()
            for batch_id, batch_examples in enumerate(train_examples_batch_iter):
                st_time = time.time()
                # get_inputs
                inputs = []
                batch_sentences = \
                    [getattr(ex, text_field) for ex in batch_examples for text_field in args.fusion_text_fields] \
                        if args.fusion_text_fields \
                        else [getattr(ex, args.text_field) for ex in batch_examples]
                batch_sentences_trimmed = None
                for tokenizer, enc_resc in zip(tokenizers, model.encoder_resources):
                    _batch_sentences = batch_sentences_trimmed if batch_sentences_trimmed else batch_sentences
                    if enc_resc.name == "bert":
                        _batch_tag = \
                            [getattr(ex, args.tag_input_label_field) for ex in batch_examples] \
                                if args.tag_input_label_field else None
                        _inputs = tokenizer.bert_subword_tokenize(_batch_sentences,
                                                                  batch_tag_sequences=_batch_tag,
                                                                  max_len=args.max_len)
                        batch_sentences_trimmed = _inputs["batch_sentences"]
                    elif enc_resc.name == "lstm":
                        if enc_resc.attr.input_representation == "char":
                            _inputs = tokenizer.char_tokenize(_batch_sentences)
                        elif enc_resc.attr.input_representation == "sc":
                            _inputs = tokenizer.sc_tokenize(_batch_sentences)
                        elif enc_resc.attr.input_representation == "fasttext":
                            _inputs = tokenizer.fasttext_tokenize(_batch_sentences)
                    inputs.append(_inputs)
                # get labels
                targets = []
                for target_label_field, target_label_vocab, task_attribute in \
                        zip(args.target_label_fields, target_label_vocabs, args.task_attributes):
                    batch_targets = [getattr(ex, target_label_field) for ex in batch_examples]
                    if task_attribute["name"] == "classification":
                        _targets = [target_label_vocab.token2idx[tgt] for tgt in batch_targets]
                    elif task_attribute["name"] == "seq_tagging":
                        _targets = [[target_label_vocab.token2idx[token] for token in tgt.split(" ")] for tgt in
                                    batch_targets]
                    targets.append(_targets)
                # run forward
                output_dict = model(inputs=inputs, targets=targets)
                # loss and backward
                loss = output_dict["loss"]
                batch_loss = loss.cpu().detach().numpy()
                train_loss += batch_loss
                # backward
                if args.grad_acc > 1:
                    loss = loss / args.grad_acc
                loss.backward()
                # optimizer step
                if (batch_id + 1) % args.grad_acc == 0 or batch_id >= n_batches - 1:
                    for optimizer in optimizers:
                        if optimizer:
                            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            optimizer.zero_grad()
                # update progress
                progress_bar(batch_id + 1, n_batches, ["batch_time", "batch_loss", "avg_batch_loss", "batch_acc"],
                             [time.time() - st_time, batch_loss, train_loss / (batch_id + 1), train_acc])
            # complete train

            """ validation """
            dev_preds, dev_true = [], []
            dev_loss, dev_acc, dev_f1 = 0., 0., 0.
            n_batches = int(np.ceil(len(dev_examples) / dev_batch_size))
            dev_examples_batch_iter = batch_iter(dev_examples, dev_batch_size, shuffle=False)
            printlog(f"len of dev data: {len(dev_examples)}")
            printlog(f"n_batches of dev data: {n_batches}")
            model.eval()
            for batch_id, batch_examples in enumerate(dev_examples_batch_iter):
                st_time = time.time()
                # get_inputs
                inputs = []
                batch_sentences = \
                    [getattr(ex, text_field) for ex in batch_examples for text_field in args.fusion_text_fields] \
                        if args.fusion_text_fields \
                        else [getattr(ex, args.text_field) for ex in batch_examples]
                batch_sentences_trimmed = None
                for tokenizer, enc_resc in zip(tokenizers, model.encoder_resources):
                    _batch_sentences = batch_sentences_trimmed if batch_sentences_trimmed else batch_sentences
                    if enc_resc.name == "bert":
                        _batch_tag = \
                            [getattr(ex, args.tag_input_label_field) for ex in batch_examples] \
                                if args.tag_input_label_field else None
                        _inputs = tokenizer.bert_subword_tokenize(_batch_sentences,
                                                                  batch_tag_sequences=_batch_tag,
                                                                  max_len=args.max_len)
                        batch_sentences_trimmed = _inputs["batch_sentences"]
                    elif enc_resc.name == "lstm":
                        if enc_resc.attr.input_representation == "char":
                            _inputs = tokenizer.char_tokenize(_batch_sentences)
                        elif enc_resc.attr.input_representation == "sc":
                            _inputs = tokenizer.sc_tokenize(_batch_sentences)
                        elif enc_resc.attr.input_representation == "fasttext":
                            _inputs = tokenizer.fasttext_tokenize(_batch_sentences)
                    inputs.append(_inputs)
                # get labels
                targets = []
                for target_label_field, target_label_vocab, task_attribute in \
                        zip(args.target_label_fields, target_label_vocabs, args.task_attributes):
                    batch_targets = [getattr(ex, target_label_field) for ex in batch_examples]
                    if task_attribute["name"] == "classification":
                        _targets = [target_label_vocab.token2idx[tgt] for tgt in batch_targets]
                    elif task_attribute["name"] == "seq_tagging":
                        _targets = [[target_label_vocab.token2idx[token] for token in tgt.split(" ")] for tgt in
                                    batch_targets]
                    targets.append(_targets)
                # run forward
                output_dict = model.predict(inputs=inputs, targets=targets)  # enhanced dict compared to forward
                # loss and accuracy
                batch_loss = output_dict["loss"].cpu().detach().numpy()
                dev_loss += batch_loss
                if not model.single_task:  # the record values only for the first module
                    dev_acc += output_dict["acc_num"][0]
                    dev_preds.extend(output_dict["preds"][0])
                    dev_true.extend(output_dict["targets"][0])
                    acc_num = output_dict["acc_num"][0]
                else:
                    dev_acc += output_dict["acc_num"]
                    dev_preds.extend(output_dict["preds"])
                    dev_true.extend(output_dict["targets"])
                    acc_num = output_dict["acc_num"]
                # update progress
                progress_bar(batch_id + 1, n_batches,
                             ["batch_time", "batch_loss", "avg_batch_loss", "batch_acc", 'avg_batch_acc'],
                             [time.time() - st_time, batch_loss, dev_loss / (batch_id + 1),
                              acc_num / dev_batch_size, dev_acc / ((batch_id + 1) * dev_batch_size)])
            # complete validation
            dev_acc /= len(dev_examples)  # exact
            dev_loss /= n_batches  # approximate
            dev_f1 = f1_score(dev_true, dev_preds, average='weighted')
            printlog("\n Validation Complete")
            printlog(f"Validation avg_loss: {dev_loss:.4f} and acc: {dev_acc:.4f}")
            printlog("\n" + classification_report(dev_true, dev_preds, digits=4))

            """ model saving """
            name = "model.pth.tar"  # "model-epoch{}.pth.tar".format(epoch_id)
            if args.checkpoint_using_accuracy:
                if (start_epoch == 0 and epoch_id == start_epoch) or best_dev_acc < dev_acc:
                    best_dev_acc, best_dev_acc_epoch = dev_acc, epoch_id
                    model.save_state_dicts(ckpt_path)
                    printlog("Model(s) saved in {} at the end of epoch(0-base) {}".format(ckpt_path, epoch_id))
                else:
                    printlog("no improvements in results to save a checkpoint")
                    printlog(f"checkpoint previously saved during epoch {best_dev_acc_epoch}(0-base) at: "
                             f"{os.path.join(ckpt_path, name)}")
            else:
                if (start_epoch == 0 and epoch_id == start_epoch) or best_dev_f1 < dev_f1:
                    best_dev_f1, best_dev_f1_epoch = dev_f1, epoch_id
                    best_dev_acc, best_dev_acc_epoch = dev_acc, epoch_id
                    model.save_state_dicts(ckpt_path)
                    printlog("Model(s) saved in {} at the end of epoch(0-base) {}".format(ckpt_path, epoch_id))
                else:
                    printlog("no improvements in results to save a checkpoint")
                    printlog(f"checkpoint previously saved during epoch {best_dev_f1_epoch}(0-base) at: "
                             f"{os.path.join(ckpt_path, name)}")

    """ testing """
    for i, selected_examples in enumerate([dev_examples, test_examples, ]):

        if selected_examples:

            """ testing on dev / test set """
            printlog("\n\n################")
            if i == 0:
                printlog(f"testing `dev` file; loading model(s) from {ckpt_path}")
            else:
                printlog(f"testing `test` file; loading model(s) from {ckpt_path}")
            printlog(f"in testing...loading model(s) from {ckpt_path}")
            model.load_state_dicts(path=ckpt_path)

            test_preds, test_true = [], []
            test_loss, test_acc, test_f1 = 0., 0., 0.
            test_batch_size = dev_batch_size
            n_batches = int(np.ceil(len(selected_examples) / test_batch_size))
            selected_examples_batch_iter = batch_iter(selected_examples, test_batch_size, shuffle=False)
            printlog(f"len of {args.mode} data: {len(selected_examples)}")
            printlog(f"n_batches of {args.mode} data: {n_batches}")
            for batch_id, batch_examples in enumerate(selected_examples_batch_iter):
                st_time = time.time()
                # forward
                # get_inputs
                inputs = []
                batch_sentences = \
                    [getattr(ex, text_field) for ex in batch_examples for text_field in args.fusion_text_fields] \
                        if args.fusion_text_fields \
                        else [getattr(ex, args.text_field) for ex in batch_examples]
                batch_sentences_trimmed = None
                for tokenizer, enc_resc in zip(tokenizers, model.encoder_resources):
                    _batch_sentences = batch_sentences_trimmed if batch_sentences_trimmed else batch_sentences
                    if enc_resc.name == "bert":
                        _batch_tag = \
                            [getattr(ex, args.tag_input_label_field) for ex in batch_examples] \
                                if args.tag_input_label_field else None
                        _inputs = tokenizer.bert_subword_tokenize(_batch_sentences,
                                                                  batch_tag_sequences=_batch_tag,
                                                                  max_len=args.max_len)
                        batch_sentences_trimmed = _inputs["batch_sentences"]
                    elif enc_resc.name == "lstm":
                        if enc_resc.attr.input_representation == "char":
                            _inputs = tokenizer.char_tokenize(_batch_sentences)
                        elif enc_resc.attr.input_representation == "sc":
                            _inputs = tokenizer.sc_tokenize(_batch_sentences)
                        elif enc_resc.attr.input_representation == "fasttext":
                            _inputs = tokenizer.fasttext_tokenize(_batch_sentences)
                    inputs.append(_inputs)
                # get labels
                targets = []
                for target_label_field, target_label_vocab, task_attribute in \
                        zip(args.target_label_fields, target_label_vocabs, args.task_attributes):
                    batch_targets = [getattr(ex, target_label_field) for ex in batch_examples]
                    if task_attribute["name"] == "classification":
                        _targets = [target_label_vocab.token2idx[tgt] for tgt in batch_targets]
                    elif task_attribute["name"] == "seq_tagging":
                        _targets = [[target_label_vocab.token2idx[token] for token in tgt.split(" ")] for tgt in
                                    batch_targets]
                    targets.append(_targets)
                # run forward
                output_dict = model.predict(inputs=inputs, targets=targets)  # enhanced dict compared to forward
                # loss and accuracy
                batch_loss = output_dict["loss"].cpu().detach().numpy()
                test_loss += batch_loss
                if not model.single_task:  # the record values only for the first module
                    test_acc += output_dict["acc_num"][0]
                    test_preds.extend(output_dict["preds"][0])
                    test_true.extend(output_dict["targets"][0])
                    acc_num = output_dict["acc_num"][0]
                else:
                    test_acc += output_dict["acc_num"]
                    test_preds.extend(output_dict["preds"])
                    test_true.extend(output_dict["targets"])
                    acc_num = output_dict["acc_num"]
                # update progress
                progress_bar(batch_id + 1, n_batches,
                             ["batch_time", "batch_loss", "avg_batch_loss", "batch_acc", 'avg_batch_acc'],
                             [time.time() - st_time, batch_loss, test_loss / (batch_id + 1),
                              acc_num / test_batch_size, test_acc / ((batch_id + 1) * test_batch_size)])
            # complete test
            test_acc /= len(selected_examples)  # exact
            test_loss /= n_batches  # approximate
            test_f1 = f1_score(test_true, test_preds, average='weighted')
            printlog("\n Test Complete")
            printlog(f"Test avg_loss: {test_loss:.4f} and acc: {test_acc:.4f}")
            printlog("\n" + classification_report(test_true, test_preds, digits=4))
            printlog("")

            # save_errors_path = os.path.join(ckpt_path, str(datetime.datetime.now()).replace(" ", "_"))
            # printlog(f"\n(NEW!) saving predictions in the folder: {save_errors_path}")
            # create_path(save_errors_path)
            # opfile = jsonlines.open(os.path.join(save_errors_path, "predictions.jsonl"), "w")
            # for i, (x, y, z) in enumerate(zip(test_exs, test_preds, test_probs)):
            #     dt = x._asdict()
            #     dt.update({"prediction": label_vocab.idx2token[y]})
            #     dt.update({"pred_probs": z})
            #     opfile.write(dt)
            # opfile.close()
            # opfile = open(os.path.join(save_errors_path, "predictions.txt"), "w")
            # for i, (x, y) in enumerate(zip(test_exs, test_preds)):
            #     try:
            #         opfile.write(f"{label_vocab.idx2token[y]} ||| {x.text}\n")
            #     except AttributeError:
            #         opfile.write(f"{label_vocab.idx2token[y]}\n")
            # opfile.close()

            # if targets is not None and len(targets) > 0:
            #     printlog(f"\n(NEW!) saving errors files in the folder: {save_errors_path}")
            #     # report
            #     report = classification_report(test_true, test_preds, digits=4,
            #                                    target_names=[label_vocab.idx2token[idx]
            #                                                  for idx in range(0, label_vocab.n_all_tokens)])
            #     printlog("\n" + report)
            #     opfile = open(os.path.join(save_errors_path, "report.txt"), "w")
            #     opfile.write(report + "\n")
            #     opfile.close()
            #     # errors
            #     opfile = jsonlines.open(os.path.join(save_errors_path, "errors.jsonl"), "w")
            #     for i, (x, y, z) in enumerate(zip(test_exs, test_preds, test_true)):
            #         if y != z:
            #             dt = x._asdict()
            #             dt.update({"prediction": label_vocab.idx2token[y]})
            #             opfile.write(dt)
            #     opfile.close()
            #     for idx_i in label_vocab.idx2token:
            #         for idx_j in label_vocab.idx2token:
            #             opfile = jsonlines.open(os.path.join(save_errors_path,
            #                                                  f"errors_pred-{idx_i}_target-{idx_j}.jsonl"), "w")
            #             temp_test_exs = [x for x, y, z in zip(test_exs, test_preds, test_true)
            #                              if (y == idx_i and z == idx_j)]
            #             for x in temp_test_exs:
            #                 dt = x._asdict()
            #                 dt.update({"prediction": label_vocab.idx2token[idx_i]})
            #                 opfile.write(dt)
            #             opfile.close()
            #     # confusion matrix
            #     cm = confusion_matrix(y_true=test_true, y_pred=test_preds, labels=list(set(test_true)))
            #     disp = ConfusionMatrixDisplay(
            #         confusion_matrix=cm,
            #         display_labels=[label_vocab.idx2token[ii] for ii in list(set(test_true))])
            #     disp = disp.plot(values_format="d")
            #     get_module_or_attr("matplotlib", "pyplot").savefig(
            #         os.path.join(save_errors_path, "confusion_matrix.png"))
            #     # plt.show()

    """ interactive """
    if args.mode == "interactive":
        printlog("\n\n################")
        printlog(f"in interactive inference mode....loading model(s) from {ckpt_path}")
        model.load_state_dicts(path=ckpt_path)

        while True:
            text_input = input("enter your text here: ")
            if text_input == "-1":
                break

            new_example = EXAMPLE(dataset=None, task=None, split_type=None, uid=None, text=text_input, text_pp=None,
                                  label=None, langids=None, seq_labels=None, langids_pp=None, meta_data=None)
            test_examples = [new_example]

            # # -----------> left unedited from previous
            # test_exs, test_preds, test_probs, test_true, targets = [], [], [], [], None
            # selected_examples = test_examples
            # n_batches = int(np.ceil(len(selected_examples) / dev_batch_size))
            # selected_examples_batch_iter = batch_iter(selected_examples, dev_batch_size, shuffle=False)
            # printlog(f"len of {args.mode} data: {len(selected_examples)}")
            # printlog(f"n_batches of {args.mode} data: {n_batches}")
            # for batch_id, batch_examples in enumerate(selected_examples_batch_iter):
            #     st_time = time.time()
            #     # forward
            #     targets = [label_vocab.token2idx[ex.label] if ex.label is not None else ex.label for ex in
            #                batch_examples]
            #     targets = None if any([x is None for x in targets]) else targets
            #     if args.model_name == "bert-lstm":
            #         batch_sentences = [getattr(ex, f"text_{args.text_field}" if args.text_field != "" else "text")
            #                            for ex in batch_examples]
            #         output_dict = model.predict(text_batch=batch_sentences, targets=targets)
            #     elif args.model_name == "bert-sc-lstm":
            #         batch_sentences = [getattr(ex, f"text_{args.text_field}" if args.text_field != "" else "text")
            #                            for ex in batch_examples]
            #         batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(
            #             batch_sentences, model.bert_tokenizer, max_len=200)
            #         batch_screps, _ = sc_tokenize(batch_sentences, word_vocab)
            #         output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
            #                                     batch_screps=batch_screps, targets=targets)
            #     elif args.model_name in ["bert-charlstm-lstm", "bert-charlstm-lstm-v2"]:
            #         batch_sentences = [getattr(ex, f"text_{args.text_field}" if args.text_field != "" else "text")
            #                            for ex in batch_examples]
            #         batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(
            #             batch_sentences, model.bert_tokenizer, max_len=200)
            #         batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
            #         output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
            #                                     batch_idxs=batch_idxs, batch_char_lengths=batch_char_lengths,
            #                                     batch_lengths=batch_lengths, targets=targets)
            #     elif args.model_name == "bert-fasttext-lstm":
            #         batch_sentences = [getattr(ex, f"text_{args.text_field}" if args.text_field != "" else "text")
            #                            for ex in batch_examples]
            #         batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(
            #             batch_sentences, model.bert_tokenizer, max_len=200)
            #         batch_embs, batch_lengths = fst_english.get_pad_vectors(
            #             batch_tokens=[line.split(" ") for line in batch_sentences],
            #             return_lengths=True)
            #         output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
            #                                     batch_screps=batch_embs, targets=targets)
            #     elif "bert" in args.model_name and args.model_name.startswith("li-"):
            #         batch_sentences = [getattr(ex, f"text_{args.text_field}" if args.text_field != "" else "text")
            #                            for ex in batch_examples]
            #         batch_tag_ids = [getattr(ex, f"langids_{args.text_field}" if args.text_field != "" else "langids")
            #                           for ex in batch_examples]
            #         # adding "other" at ends
            #         batch_tag_ids = [" ".join([str(lid_label_vocab.sos_token_idx)] +
            #                                    [str(lid_label_vocab.token2idx[lang]) for lang in lang_ids.split(" ")] +
            #                                    [str(lid_label_vocab.eos_token_idx)])
            #                           for lang_ids in batch_tag_ids]
            #         output_dict = model.predict(batch_sentences, batch_tag_ids, targets=targets)
            #     elif "bert" in args.model_name and args.model_name.startswith("posi-"):
            #         batch_sentences = [getattr(ex, f"text_{args.text_field}" if args.text_field != "" else "text")
            #                            for ex in batch_examples]
            #         batch_pos_ids = [getattr(ex, f"postags_{args.text_field}" if args.text_field != "" else "postags")
            #                          for ex in batch_examples]
            #         # adding "other" at ends
            #         batch_pos_ids = [" ".join([str(pos_label_vocab.sos_token_idx)] +
            #                                   [str(pos_label_vocab.token2idx[lang]) for lang in pos_ids.split(" ")] +
            #                                   [str(pos_label_vocab.eos_token_idx)])
            #                          for pos_ids in batch_pos_ids]
            #         output_dict = model.predict(batch_sentences, batch_pos_ids, targets=targets)
            #     elif args.model_name == "bert-semantic-similarity":
            #         batch_sentences = []
            #         for text_field in ["src", "tgt"]:
            #             batch_sentences.extend([getattr(ex, text_field) for ex in batch_examples])
            #         output_dict = model.predict(text_batch=batch_sentences, targets=targets)
            #     elif "bert" in args.model_name:
            #         if args.fusion_text_fields:
            #             batch_sentences = []
            #             for text_field in args.fusion_text_fields:
            #                 batch_sentences.extend([getattr(ex, text_field) for ex in batch_examples])
            #             output_dict = model.predict(text_batch=batch_sentences, targets=targets)
            #         elif args.multitask_lid_sa:
            #             batch_sentences = [getattr(ex, f"text_{args.text_field}" if args.text_field != "" else "text")
            #                                for ex in batch_examples]
            #             lid_targets = [[lid_label_vocab.token2idx[token] for token in
            #                             getattr(ex,
            #                                     f"langids_{args.target_label_fields}" if args.target_label_fields != "" else "langids").split(
            #                                 " ")]
            #                            for ex in batch_examples]
            #             output_dict = model.predict(text_batch=batch_sentences, sa_targets=targets,
            #                                         lid_targets=lid_targets)
            #         elif args.sentence_bert:
            #             batch_sentences = [getattr(ex, f"text_{args.text_field}" if args.text_field != "" else "text")
            #                                for ex in batch_examples]
            #             output_dict = model.predict(text_batch=batch_sentences, targets=targets)
            #         else:
            #             batch_sentences = [getattr(ex, f"text_{args.text_field}" if args.text_field != "" else "text")
            #                                for ex in batch_examples]
            #             output_dict = model.predict(text_batch=batch_sentences, targets=targets)
            #     elif args.model_name == "fasttext-vanilla":
            #         # batch_english = [" ".join([token for token, tag in zip(ex.text_trt.split(" "), ex.langids_pp.split(" "))
            #         #                            if tag.lower() != "hin"]) for ex in batch_examples]
            #         # input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
            #         # batch_hindi = [" ".join([token for token, tag in zip(ex.text_trt.split(" "), ex.langids_pp.split(" "))
            #         #                          if tag.lower() == "hin"]) for ex in batch_examples]
            #         # input_hindi = torch.tensor(fst_hindi.get_phrase_vector(batch_english))
            #         # output_dict = model.predict(input1=input_english, input2=input_hindi, targets=targets)
            #         batch_english = [
            #             " ".join([token for token, tag in zip(ex.text_pp.split(" "), ex.langids_pp.split(" "))])
            #             for ex in batch_examples]
            #         input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
            #         output_dict = model.predict(input1=input_english, targets=targets)
            #     elif args.model_name == "charlstmlstm":
            #         batch_sentences = [getattr(ex, f"text_{args.text_field}" if args.text_field != "" else "text")
            #                            for ex in batch_examples]
            #         batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
            #         output_dict = model.predict(batch_idxs, batch_char_lengths, batch_lengths, targets=targets)
            #     elif args.model_name == "sclstm":
            #         batch_sentences = [getattr(ex, f"text_{args.text_field}" if args.text_field != "" else "text")
            #                            for ex in batch_examples]
            #         batch_screps, batch_lengths = sc_tokenize(batch_sentences, word_vocab)
            #         output_dict = model.predict(batch_screps, batch_lengths, targets=targets)
            #     elif args.model_name == "fasttext-lstm":
            #         batch_sentences = [getattr(ex, f"text_{args.text_field}" if args.text_field != "" else "text")
            #                            for ex in batch_examples]
            #         batch_embs, batch_lengths = fst_english.get_pad_vectors(
            #             batch_tokens=[line.split(" ") for line in batch_sentences],
            #             return_lengths=True)
            #         output_dict = model.predict(batch_embs, batch_lengths, targets=targets)
            #     test_exs.extend(batch_examples)
            #     test_preds.extend(output_dict["preds"])
            #     test_probs.extend(output_dict["probs"])
            #     if targets is not None:
            #         test_true.extend(targets)
            #     # update progress
            #     # progress_bar(batch_id + 1, n_batches, ["batch_time"], [time.time() - st_time])
            # # <-----------
            # printlog([label_vocab.idx2token[y] for y in test_preds])

    if (not args.debug) and logger_file_name:
        os.system(f"cat {logger_file_name}")
