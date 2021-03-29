import argparse
import datetime
import json
import logging
import os
import sys
import time

import jsonlines
import numpy as np
import torch
from pytorch_pretrained_bert import BertAdam
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from .helpers import create_vocab, load_vocab, char_tokenize, sc_tokenize, bert_subword_tokenize
from .helpers import get_model_nparams, batch_iter, progress_bar, FastTextVecs
from .models import CNNCharacterBertMLP
from .models import SentenceBertForSemanticSimilarity
from .models import SentenceTransformersBertMLP
from .models import SimpleMLP, CharLstmLstmMLP, ScLstmMLP
from .models import WholeWordBertLstmMLP, WholeWordBertScLstmMLP, WholeWordBertCharLstmLstmMLP
from .models import WholeWordBertMLP, WholeWordBertForSeqClassificationAndTagging, FusedBertMLP, SentenceBert
from .models import WholeWordBertXXXInformedMLP
from ..datasets import read_datasets_jsonl, create_path, EXAMPLE
from ..extras import get_module_or_attr
from ..paths import SRC_ROOT_PATH

DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
LOGS_FOLDER = "./logs"
MODEL_NAME_CHOICES_AVAILABLE = [
    "xlm-roberta-base", "bert-base-cased", "bert-base-multilingual-cased",  # other hgface models,
    "fasttext-vanilla", "fasttext-lstm", "charlstmlstm", "sclstm",
    "bert-lstm", "bert-fasttext-lstm", "bert-sc-lstm", "bert-charlstm-lstm",  # "bert-charlstm-lstm-v2",
    "li-bert-base-cased", "li-xlm-roberta-base", "posi-xlm-roberta-base",
    "bert-semantic-similarity",
    "cnn-character-bert"
]
FUSION_STRATEGY_CHOICES = ["concat", "mean_pool", "max_pool"]
printlog = print


class Args:

    def __init__(self, **kwargs):
        self.dataset_folder = kwargs.get("dataset_folder")  # folder path for train, test, dev
        self.mode = kwargs.get("mode", "train_dev_test")  # to just train or to train and infer on dev and test sets

        self.augment_train_datasets = kwargs.get("augment_train_datasets", "")
        self.text_type = kwargs.get("text_type", "")  # "" implies `text_raw` field
        self.langids_type = kwargs.get("langids_type", None)

        self.model_name = kwargs.get("model_name")
        # to copy params from a custom trained model, intelligently loads relevant params
        self.custom_pretrained_path = kwargs.get("custom_pretrained_path", None)
        # self.override_model_load_path = kwargs.get("override_model_load_path", None)  # hgface direct load using auto
        self.max_epochs = kwargs.get("max_epochs", 5)
        self.patience = kwargs.get("patience", 4)
        self.batch_size = kwargs.get("batch_size", None)

        self.eval_ckpt_path = kwargs.get("eval_ckpt_path", None)
        self.save_errors_path = kwargs.get("save_errors_path", None)
        self.checkpoint_using_accuracy = kwargs.get("checkpoint_using_accuracy", False)
        self.debug = kwargs.get("debug", False)

        """ extras """
        self.fusion_strategy = kwargs.get("fusion_strategy", None)
        # TODO: Expects full text name. ex: --fusion-text-types text text_hi text_trt text_en
        self.fusion_text_types = kwargs.get("fusion_text_types", [])
        self.multitask_lid_sa = kwargs.get("multitask_lid_sa", False)
        self.sentence_bert = kwargs.get("sentence_bert", False)  # our version
        self.sentence_transformers = kwargs.get("sentence_transformers", False)  # sbert library


def _get_checkpoints_path(args):
    if args.eval_ckpt_path:
        assert any([ii in args.mode for ii in ["dev", "test", "interactive"]])
        CHECKPOINT_PATH = args.eval_ckpt_path
    else:
        assert "train" in args.mode, print("--mode must contain `train` if no eval ckpt path is specified")
        CHECKPOINT_PATH = os.path.join(args.dataset_folder, "checkpoints",
                                       f'{str(datetime.datetime.now()).replace(" ", "_")}')
        if os.path.exists(CHECKPOINT_PATH):
            # subparts = [x for x in CHECKPOINT_PATH.split("/") if x]
            # subparts[-1] = subparts[-1] + "--" + str(datetime.datetime.now()).replace(" ", "_")
            # CHECKPOINT_PATH = "/".join(subparts)
            raise Exception(
                f"CHECKPOINT_PATH: {CHECKPOINT_PATH} already exists. Did you mean to set mode to dev or test?")
        create_path(CHECKPOINT_PATH)
    printlog(f"CHECKPOINT_PATH: {CHECKPOINT_PATH}")
    return CHECKPOINT_PATH


def _set_logger(args, CHECKPOINT_PATH):
    global printlog

    if not args.debug:
        if not os.path.exists(os.path.join(CHECKPOINT_PATH, LOGS_FOLDER)):
            os.makedirs(os.path.join(CHECKPOINT_PATH, LOGS_FOLDER))
        # logger_file_name = os.path.join(CHECKPOINT_PATH, LOGS_FOLDER, "{}_{}".format(
        #     os.path.basename(__file__).split('.')[-2], args.model_name))
        logger_file_name = os.path.join(CHECKPOINT_PATH, LOGS_FOLDER,
                                        f'{str(datetime.datetime.now()).replace(" ", "_")}')
        logging.basicConfig(level=logging.INFO, filename=logger_file_name, filemode='a',
                            datefmt='%Y-%m-%d:%H:%M:%S',
                            format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s')
        printlog = logging.info
    else:
        logger_file_name = None
        printlog = print
    printlog('\n\n\n\n--------------------------------\nBeginning to log...\n--------------------------------\n')
    printlog(" ".join(sys.argv))
    return logger_file_name


def _validate_args(args):
    def _validate_fusion_strategy(name):
        if name and name not in FUSION_STRATEGY_CHOICES:
            raise NotImplementedError(f"choice of fusion startegy - `{name}` is unknown")
        return

    def _validate_model_name(name):
        if name not in MODEL_NAME_CHOICES_AVAILABLE:
            raise NotImplementedError(f"choice of model name - `{name}` is unknown")
        return

    _validate_model_name(args.model_name)
    _validate_fusion_strategy(args.fusion_strategy)

    if args.batch_size:
        assert args.batch_size > 0, printlog(f"args.batch_size: {args.batch_size} must be a positive integer")
    if "train" not in args.mode and args.model_name is not None:
        assert "bert" in args.model_name or "fasttext" in args.model_name or "lstm" in args.model_name
    if args.model_name in ["fasttext-vanilla", "bert-semantic-similarity"]:
        printlog(f"dropping text_type as it is irrelevant with model_name=={args.model_name}")
        args.text_type = None
    if args.fusion_text_types:
        printlog("dropping text_type as it is irrelevant with fusion_text_types")
        args.text_type = ""
    if args.multitask_lid_sa and args.fusion_text_types:
        raise Exception
    args.langids_type = args.text_type
    printlog("dropping your inputted info about langids_type and setting it to same as text_type")

    return


def _load_train_examples(args):
    train_examples = read_datasets_jsonl(os.path.join(args.dataset_folder, "train.jsonl"), "train")
    if args.text_type is not None and "+" in args.text_type:  # eg. "pp+trt", "+noisy"
        new_train_examples = []
        types = args.text_type.split("+")
        for ex in train_examples:
            for typ in types:
                new_example = EXAMPLE(dataset=ex.dataset, task=ex.task, split_type=ex.split_type,
                                      uid=ex.uid + (f"_{typ}" if typ != "" else f"_raw"),
                                      text=getattr(ex, f"text_{typ}" if typ != "" else "text"),
                                      text_pp=ex.text_pp, label=ex.label, langids=ex.langids,
                                      seq_labels=None, langids_pp=None, meta_data=None)
                new_train_examples.append(new_example)
        train_examples = new_train_examples
        printlog(f"train examples increased to {len(train_examples)} due to --text-type {args.text_type}")
        args.text_type = ""  # due to already inclusion of other data in training
    if args.augment_train_datasets:
        train_augment_examples = []
        dnames = [x.strip() for x in args.augment_train_datasets.split(",")]
        for name in dnames:
            temp_examples = read_datasets_jsonl(os.path.join(name, "train.jsonl"), "train")
            train_augment_examples.extend(temp_examples)
        printlog(f"obtained {len(train_augment_examples)} additional train examples")
        train_examples.extend(train_augment_examples)
    return train_examples


def run_classification(args: argparse.Namespace = None, **kwargs):
    args = args or Args(**kwargs)

    """ basics """
    CHECKPOINT_PATH = _get_checkpoints_path(args)
    logger_file_name = _set_logger(args, CHECKPOINT_PATH)
    _validate_args(args)

    """ settings """
    start_epoch, n_epochs = 0, args.max_epochs
    train_batch_size, dev_batch_size = (args.batch_size, args.batch_size) if args.batch_size \
        else (16, 16) if "bert" in args.model_name else (64, 64)
    grad_acc = 2 if "bert" in args.model_name else 1
    if args.fusion_text_types:
        fusion_n = len(args.fusion_text_types)
        train_batch_size, dev_batch_size = int(train_batch_size / fusion_n), int(dev_batch_size / fusion_n)
        grad_acc *= fusion_n
    if args.model_name == "bert-semantic-similarity":
        fusion_n = 2
        train_batch_size, dev_batch_size = int(train_batch_size / fusion_n), int(dev_batch_size / fusion_n)
        grad_acc *= fusion_n

    """ load dataset """
    train_examples, dev_examples, test_examples = [], [], []
    all_modes = args.mode.split("_")
    assert len(all_modes), printlog("Expected at least one mode")
    for mode in all_modes:
        if "train" in mode:  # "train", "train-fb", "train-fb+twitter", etc.
            train_examples = _load_train_examples(args)
        elif "dev" in mode:
            dev_examples = read_datasets_jsonl(os.path.join(args.dataset_folder, f"{mode}.jsonl"), f"dev")
        elif "test" in mode:  # can be anything like "test", "test-xyz", "test-collate", etc.
            test_examples = read_datasets_jsonl(os.path.join(args.dataset_folder, f"{mode}.jsonl"), f"test")
        else:
            raise ValueError(f"invalid mode `{mode}` encountered in {args.mode}")
    if args.debug:
        printlog("debug mode enabled ...")
        train_examples = train_examples[:40]
        dev_examples = dev_examples[:20]
        test_examples = test_examples[:20]

    """ check dataset """
    # # TODO: Move this condition to dataset creation time
    # if args.multitask_lid_sa:
    #     examples = None
    #     if args.mode == 'train':
    #         examples = train_examples + dev_examples
    #     elif args.mode == 'test':
    #         examples = test_examples
    #     for ex in examples:
    #         if len(getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text").split(" ")) != len(
    #                 getattr(ex, f"text_{args.langids_type}" if args.langids_type != "" else "langids").split(" ")):
    #             raise AssertionError

    """ obtain vocab(s) """
    label_vocab, lid_label_vocab, pos_label_vocab, word_vocab = None, None, None, None
    if "train" in args.mode:
        label_vocab = create_vocab([ex.label for ex in train_examples], is_label=True)
        if args.multitask_lid_sa or args.model_name.startswith("li-"):
            lid_label_vocab = create_vocab(
                [i for ex in train_examples for i in getattr(ex, f"langids_{args.langids_type}"
                if args.langids_type != "" else "langids").split(" ")], is_label=False)
        if args.model_name.startswith("posi-"):
            pos_label_vocab = create_vocab([i for ex in train_examples for i in getattr(ex, f"postags_{args.text_type}"
            if args.text_type != "" else "postags").split(" ")], is_label=False)
        if any([term in args.model_name for term in ["lstm", ]]):
            word_vocab = create_vocab(
                [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text") for ex in train_examples],
                is_label=False,
                load_char_tokens=True)
    else:
        label_vocab = load_vocab(os.path.join(CHECKPOINT_PATH, "label_vocab.json"))
        if args.multitask_lid_sa or args.model_name.startswith("li-"):
            lid_label_vocab = load_vocab(os.path.join(CHECKPOINT_PATH, "lid_label_vocab.json"))
        if args.model_name.startswith("posi-"):
            pos_label_vocab = load_vocab(os.path.join(CHECKPOINT_PATH, "pos_label_vocab.json"))
        if any([term in args.model_name for term in ["lstm", ]]):
            word_vocab = load_vocab(os.path.join(CHECKPOINT_PATH, "word_vocab.json"))

    """ define and initialize model """
    model = None
    if args.sentence_transformers:
        pretrained_path = args.model_name
        model = SentenceTransformersBertMLP(out_dim=label_vocab.n_all_tokens,
                                            pretrained_path=pretrained_path,
                                            finetune_bert=True)
    elif args.model_name == "bert-lstm":
        printlog("bert variant used in bert-lstm is xlm-roberta-base")
        model = WholeWordBertLstmMLP(out_dim=label_vocab.n_all_tokens, pretrained_path="xlm-roberta-base",
                                     finetune_bert=True)
    elif args.model_name == "bert-sc-lstm":
        printlog("bert variant used in bert-sc-lstm is xlm-roberta-base")
        model = WholeWordBertScLstmMLP(screp_dim=3 * len(word_vocab.chartoken2idx), out_dim=label_vocab.n_all_tokens,
                                       pretrained_path="xlm-roberta-base", finetune_bert=True)
    elif args.model_name == "bert-charlstm-lstm":
        printlog("bert variant used in bert-charlstm-lstm is xlm-roberta-base")
        model = WholeWordBertCharLstmLstmMLP(nchars=len(word_vocab.chartoken2idx),
                                             char_emb_dim=128,
                                             char_padding_idx=word_vocab.char_pad_token_idx,
                                             out_dim=label_vocab.n_all_tokens,
                                             pretrained_path="xlm-roberta-base")
    elif args.model_name == "bert-charlstm-lstm-v2":
        printlog("bert variant used in bert-charlstm-lstm-v2 is xlm-roberta-base")
        assert os.path.exists(args.custom_pretrained_path)
        model = WholeWordBertCharLstmLstmMLP(nchars=len(word_vocab.chartoken2idx),
                                             char_emb_dim=128,
                                             char_padding_idx=word_vocab.char_pad_token_idx,
                                             out_dim=label_vocab.n_all_tokens,
                                             pretrained_path="xlm-roberta-base",
                                             freezable_pretrained_path=args.custom_pretrained_path,
                                             device=DEVICE)
        args.custom_pretrained_path = ""
    elif args.model_name == "bert-fasttext-lstm":
        # load pretrained
        fst_english = FastTextVecs("en")
        printlog("Loaded en fasttext model")
        printlog("bert variant used in bert-fasttext-lstm is xlm-roberta-base")
        model = WholeWordBertScLstmMLP(screp_dim=fst_english.ft_dim, out_dim=label_vocab.n_all_tokens,
                                       pretrained_path="xlm-roberta-base", finetune_bert=True)
    elif "bert" in args.model_name and args.model_name.startswith("li-"):
        assert os.path.exists(args.custom_pretrained_path)
        model = WholeWordBertXXXInformedMLP(out_dim=label_vocab.n_all_tokens,
                                            pretrained_path=args.custom_pretrained_path,
                                            n_lang_ids=lid_label_vocab.n_all_tokens,
                                            device=DEVICE,
                                            token_type_pad_idx=lid_label_vocab.pad_token_idx)
        args.custom_pretrained_path = ""
    elif "bert" in args.model_name and args.model_name.startswith("posi-"):
        assert os.path.exists(args.custom_pretrained_path)
        model = WholeWordBertXXXInformedMLP(out_dim=label_vocab.n_all_tokens,
                                            pretrained_path=args.custom_pretrained_path,
                                            n_lang_ids=pos_label_vocab.n_all_tokens, device=DEVICE,
                                            token_type_pad_idx=pos_label_vocab.pad_token_idx)
        args.custom_pretrained_path = ""
    elif args.model_name == "bert-semantic-similarity":
        printlog("bert variant used in bert-semantic-similarity is xlm-roberta-base")
        model = SentenceBertForSemanticSimilarity(out_dim=label_vocab.n_all_tokens, pretrained_path="xlm-roberta-base",
                                                  finetune_bert=True)
    elif args.model_name == "cnn-character-bert":
        model = CNNCharacterBertMLP(out_dim=label_vocab.n_all_tokens,
                                    pretrained_path="bert-base-uncased",
                                    finetune_bert=True)
    elif "bert" in args.model_name:
        pretrained_path = args.model_name
        if args.multitask_lid_sa:
            model = WholeWordBertForSeqClassificationAndTagging(sent_out_dim=label_vocab.n_all_tokens,
                                                                lang_out_dim=lid_label_vocab.n_all_tokens,
                                                                pretrained_path=pretrained_path)
        elif args.fusion_text_types:
            model = FusedBertMLP(out_dim=label_vocab.n_all_tokens, pretrained_path=pretrained_path,
                                 finetune_bert=True, fusion_n=fusion_n, fusion_strategy=args.fusion_strategy)
        elif args.sentence_bert:
            model = SentenceBert(out_dim=label_vocab.n_all_tokens, pretrained_path=pretrained_path,
                                 finetune_bert=True)
        else:
            # if args.override_model_load_path:
            #     pretrained_path = args.override_model_load_path
            model = WholeWordBertMLP(out_dim=label_vocab.n_all_tokens, pretrained_path=pretrained_path,
                                     finetune_bert=True)
    elif args.model_name == "fasttext-vanilla":
        # load pretrained
        fst_english = FastTextVecs("en")
        printlog("Loaded en fasttext model")
        fst_hindi = FastTextVecs("hi")
        printlog("Loaded hi fasttext model")
        # define model
        #   choose model based on if you want to pass en and hi token details seperately, or just only one of en or hi
        model = SimpleMLP(out_dim=label_vocab.n_all_tokens,
                          input_dim1=fst_english.ft_dim)  # input_dim2=fst_hindi.ft_dim)
    elif args.model_name == "charlstmlstm":
        model = CharLstmLstmMLP(nchars=len(word_vocab.chartoken2idx),
                                char_emb_dim=128,
                                char_padding_idx=word_vocab.char_pad_token_idx,
                                padding_idx=word_vocab.pad_token_idx,
                                output_dim=label_vocab.n_all_tokens)
    elif args.model_name == "sclstm":
        model = ScLstmMLP(screp_dim=3 * len(word_vocab.chartoken2idx),
                          padding_idx=word_vocab.pad_token_idx,
                          output_dim=label_vocab.n_all_tokens)
    elif args.model_name == "fasttext-lstm":
        # load pretrained
        fst_english = FastTextVecs("en")
        printlog("Loaded en fasttext model")
        model = ScLstmMLP(screp_dim=fst_english.ft_dim,
                          padding_idx=word_vocab.pad_token_idx,
                          output_dim=label_vocab.n_all_tokens)

    if "bert" in args.model_name and "train" in args.mode and args.custom_pretrained_path:
        printlog(f"\nLoading weights from args.custom_pretrained_path:{args.custom_pretrained_path}")
        pretrained_dict = torch.load(f"{args.custom_pretrained_path}/pytorch_model.bin",
                                     map_location=torch.device(DEVICE))
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        used_dict = {}
        for k, v in model_dict.items():
            if "classifier.weight" in k or "classifier.bias" in k:
                printlog(f"Ignoring to load '{k}' from custom pretrained model")
                continue
            if k in pretrained_dict and v.shape == pretrained_dict[k].shape:
                used_dict[k] = pretrained_dict[k]
            elif ".".join(k.split(".")[1:]) in pretrained_dict and v.shape == pretrained_dict[
                ".".join(k.split(".")[1:])].shape:
                used_dict[k] = pretrained_dict[".".join(k.split(".")[1:])]
            elif "bert." + ".".join(k.split(".")[1:]) in pretrained_dict and v.shape == pretrained_dict[
                "bert." + ".".join(k.split(".")[1:])].shape:
                used_dict[k] = pretrained_dict["bert." + ".".join(k.split(".")[1:])]
            elif "bert." + ".".join(k.split(".")[3:]) in pretrained_dict and v.shape == pretrained_dict[
                "bert." + ".".join(k.split(".")[3:])].shape:
                used_dict[k] = pretrained_dict["bert." + ".".join(k.split(".")[3:])]
            elif "roberta." + ".".join(k.split(".")[1:]) in pretrained_dict and v.shape == pretrained_dict[
                "roberta." + ".".join(k.split(".")[1:])].shape:
                used_dict[k] = pretrained_dict["roberta." + ".".join(k.split(".")[1:])]
            elif "bert." + k in pretrained_dict and v.shape == pretrained_dict["bert." + k].shape:
                used_dict[k] = pretrained_dict["bert." + k]
            elif "roberta." + k in pretrained_dict and v.shape == pretrained_dict["roberta." + k].shape:
                used_dict[k] = pretrained_dict["roberta." + k]
        unused_dict = {k: v for k, v in model_dict.items() if k not in used_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(used_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        # 4. printlog unused_dict
        printlog("WARNING !!!")
        printlog(
            f"Following {len([*unused_dict.keys()])} keys are not updated from {args.custom_pretrained_path}/pytorch_model.bin")
        printlog(f"  →→ {[*unused_dict.keys()]}")

    printlog(f"number of parameters (all, trainable) in your model: {get_model_nparams(model)}")
    model.to(DEVICE)

    """ define optimizer """
    if "train" in args.mode:
        if "bert" in args.model_name and "lstm" in args.model_name:
            bert_model_params_names = ["bert_model." + x[0] for x in model.bert_model.named_parameters()]
            # others
            other_params = [param[1] for param in list(model.named_parameters()) if
                            param[0] not in bert_model_params_names]
            printlog(f"{len(other_params)} number of params are being optimized with Adam")
            optimizer = torch.optim.Adam(other_params, lr=0.001)
            # bert
            bert_params = [param for param in list(model.named_parameters()) if param[0] in bert_model_params_names]
            param_optimizer = bert_params
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            t_total = int(len(train_examples) / train_batch_size / grad_acc * n_epochs)
            lr = 2e-5  # 1e-4 or 2e-5 or 5e-5
            bert_optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.1, t_total=t_total)
            printlog(f"{len(bert_params)} number of params are being optimized with BertAdam")
        elif "bert" in args.model_name or args.sentence_transformers:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            t_total = int(len(train_examples) / train_batch_size / grad_acc * n_epochs)
            lr = 2e-5  # 1e-4 or 2e-5 or 5e-5
            optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.1, t_total=t_total)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
            model.zero_grad()
            model.train()
            for batch_id, batch_examples in enumerate(train_examples_batch_iter):
                st_time = time.time()
                # forward
                targets = [label_vocab.token2idx[ex.label] for ex in batch_examples]
                if args.sentence_transformers:
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    output_dict = model(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "bert-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    output_dict = model(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "bert-sc-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_screps, _ = sc_tokenize(batch_sentences, word_vocab)
                    output_dict = model(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                        batch_screps=batch_screps, targets=targets)
                elif args.model_name in ["bert-charlstm-lstm", "bert-charlstm-lstm-v2"]:
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
                    output_dict = model(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                        batch_idxs=batch_idxs, batch_char_lengths=batch_char_lengths,
                                        batch_lengths=batch_lengths, targets=targets)
                elif args.model_name == "bert-fasttext-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_embs, batch_lengths = fst_english.get_pad_vectors(
                        batch_tokens=[line.split(" ") for line in batch_sentences],
                        return_lengths=True)
                    output_dict = model(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                        batch_screps=batch_embs, targets=targets)
                elif "bert" in args.model_name and args.model_name.startswith("li-"):
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_lang_ids = [getattr(ex, f"langids_{args.text_type}" if args.text_type != "" else "langids")
                                      for ex in batch_examples]
                    # adding "other" at ends
                    batch_lang_ids = [" ".join([str(lid_label_vocab.sos_token_idx)] +
                                               [str(lid_label_vocab.token2idx[lang]) for lang in lang_ids.split(" ")] +
                                               [str(lid_label_vocab.eos_token_idx)])
                                      for lang_ids in batch_lang_ids]
                    output_dict = model(batch_sentences, batch_lang_ids, targets=targets)
                elif "bert" in args.model_name and args.model_name.startswith("posi-"):
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_pos_ids = [getattr(ex, f"postags_{args.text_type}" if args.text_type != "" else "postags")
                                     for ex in batch_examples]
                    # adding "other" at ends
                    batch_pos_ids = [" ".join([str(pos_label_vocab.sos_token_idx)] +
                                              [str(pos_label_vocab.token2idx[lang]) for lang in pos_ids.split(" ")] +
                                              [str(pos_label_vocab.eos_token_idx)])
                                     for pos_ids in batch_pos_ids]
                    output_dict = model(batch_sentences, batch_pos_ids, targets=targets)
                elif args.model_name == "bert-semantic-similarity":
                    batch_sentences = []
                    for text_type in ["src", "tgt"]:
                        batch_sentences.extend([getattr(ex, text_type) for ex in batch_examples])
                    output_dict = model(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "cnn-character-bert":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    output_dict = model(text_batch=batch_sentences, targets=targets)
                elif "bert" in args.model_name:
                    if args.fusion_text_types:
                        batch_sentences = []
                        for text_type in args.fusion_text_types:
                            batch_sentences.extend([getattr(ex, text_type) for ex in batch_examples])
                        output_dict = model(text_batch=batch_sentences, targets=targets)
                    elif args.multitask_lid_sa:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        lid_targets = [[lid_label_vocab.token2idx[token] for token in
                                        getattr(ex,
                                                f"langids_{args.langids_type}" if args.langids_type != "" else "langids").split(
                                            " ")]
                                       for ex in batch_examples]
                        output_dict = model(text_batch=batch_sentences, sa_targets=targets, lid_targets=lid_targets)
                    elif args.sentence_bert:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        output_dict = model(text_batch=batch_sentences, targets=targets)
                    else:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        output_dict = model(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "fasttext-vanilla":
                    # batch_english = [" ".join([token for token, tag in zip(ex.text_trt.split(" "),
                    #                                                        ex.langids_pp.split(" "))
                    #                            if tag.lower() != "hin"]) for ex in batch_examples]
                    # input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
                    # batch_hindi = [" ".join([token for token, tag in zip(ex.text_trt.split(" "), ex.langids_pp.split(" "))
                    #                          if tag.lower() == "hin"]) for ex in batch_examples]
                    # input_hindi = torch.tensor(fst_hindi.get_phrase_vector(batch_english))
                    # output_dict = model(input1=input_english, input2=input_hindi, targets=targets)
                    batch_english = [
                        " ".join([token for token, tag in zip(ex.text_pp.split(" "), ex.langids_pp.split(" "))])
                        for ex in batch_examples]
                    input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
                    output_dict = model(input1=input_english, targets=targets)
                elif args.model_name == "charlstmlstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
                    output_dict = model(batch_idxs, batch_char_lengths, batch_lengths, targets=targets)
                elif args.model_name == "sclstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_screps, batch_lengths = sc_tokenize(batch_sentences, word_vocab)
                    output_dict = model(batch_screps, batch_lengths, targets=targets)
                elif args.model_name == "fasttext-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_embs, batch_lengths = fst_english.get_pad_vectors(
                        batch_tokens=[line.split(" ") for line in batch_sentences],
                        return_lengths=True)
                    output_dict = model(batch_embs, batch_lengths, targets=targets)
                loss = output_dict["loss"]
                batch_loss = loss.cpu().detach().numpy()
                train_loss += batch_loss
                # backward
                if grad_acc > 1:
                    loss = loss / grad_acc
                loss.backward()
                # optimizer step
                if (batch_id + 1) % grad_acc == 0 or batch_id >= n_batches - 1:
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if args.model_name in ["bert-lstm", "bert-sc-lstm", "bert-charlstm-lstm"]:
                        bert_optimizer.step()
                    optimizer.step()
                    model.zero_grad()
                # update progress
                progress_bar(batch_id + 1, n_batches, ["batch_time", "batch_loss", "avg_batch_loss", "batch_acc"],
                             [time.time() - st_time, batch_loss, train_loss / (batch_id + 1), train_acc])
                # break
            printlog("")

            """ validation """
            dev_preds, dev_true = [], []
            dev_loss, dev_acc, dev_f1 = 0., 0., 0.
            n_batches = int(np.ceil(len(dev_examples) / dev_batch_size))
            dev_examples_batch_iter = batch_iter(dev_examples, dev_batch_size, shuffle=False)
            printlog(f"len of dev data: {len(dev_examples)}")
            printlog(f"n_batches of dev data: {n_batches}")
            for batch_id, batch_examples in enumerate(dev_examples_batch_iter):
                st_time = time.time()
                # forward
                targets = [label_vocab.token2idx[ex.label] for ex in batch_examples]
                if args.sentence_transformers:
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "bert-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "bert-sc-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_screps, _ = sc_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                                batch_screps=batch_screps, targets=targets)
                elif args.model_name in ["bert-charlstm-lstm", "bert-charlstm-lstm-v2"]:
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                                batch_idxs=batch_idxs, batch_char_lengths=batch_char_lengths,
                                                batch_lengths=batch_lengths, targets=targets)
                elif args.model_name == "bert-fasttext-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_embs, batch_lengths = fst_english.get_pad_vectors(
                        batch_tokens=[line.split(" ") for line in batch_sentences],
                        return_lengths=True)
                    output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                                batch_screps=batch_embs, targets=targets)
                elif "bert" in args.model_name and args.model_name.startswith("li-"):
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_lang_ids = [getattr(ex, f"langids_{args.text_type}" if args.text_type != "" else "langids")
                                      for ex in batch_examples]
                    # adding "other" at ends
                    batch_lang_ids = [" ".join([str(lid_label_vocab.sos_token_idx)] +
                                               [str(lid_label_vocab.token2idx[lang]) for lang in lang_ids.split(" ")] +
                                               [str(lid_label_vocab.eos_token_idx)])
                                      for lang_ids in batch_lang_ids]
                    output_dict = model.predict(batch_sentences, batch_lang_ids, targets=targets)
                elif "bert" in args.model_name and args.model_name.startswith("posi-"):
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_pos_ids = [getattr(ex, f"postags_{args.text_type}" if args.text_type != "" else "postags")
                                     for ex in batch_examples]
                    # adding "other" at ends
                    batch_pos_ids = [" ".join([str(pos_label_vocab.sos_token_idx)] +
                                              [str(pos_label_vocab.token2idx[lang]) for lang in pos_ids.split(" ")] +
                                              [str(pos_label_vocab.eos_token_idx)])
                                     for pos_ids in batch_pos_ids]
                    output_dict = model.predict(batch_sentences, batch_pos_ids, targets=targets)
                elif args.model_name == "bert-semantic-similarity":
                    batch_sentences = []
                    for text_type in ["src", "tgt"]:
                        batch_sentences.extend([getattr(ex, text_type) for ex in batch_examples])
                    output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "cnn-character-bert":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif "bert" in args.model_name:
                    if args.fusion_text_types:
                        batch_sentences = []
                        for text_type in args.fusion_text_types:
                            batch_sentences.extend([getattr(ex, text_type) for ex in batch_examples])
                        output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                    elif args.multitask_lid_sa:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        lid_targets = [[lid_label_vocab.token2idx[token] for token in
                                        getattr(ex,
                                                f"langids_{args.langids_type}" if args.langids_type != "" else "langids").split(
                                            " ")]
                                       for ex in batch_examples]
                        output_dict = model.predict(text_batch=batch_sentences, sa_targets=targets,
                                                    lid_targets=lid_targets)
                    elif args.sentence_bert:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                    else:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "fasttext-vanilla":
                    # batch_english = [" ".join([token for token, tag in zip(ex.text_trt.split(" "),
                    #                                                        ex.langids_pp.split(" "))
                    #                            if tag.lower() != "hin"]) for ex in batch_examples]
                    # input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
                    # batch_hindi = [" ".join([token for token, tag in zip(ex.text_trt.split(" "), ex.langids_pp.split(" "))
                    #                          if tag.lower() == "hin"]) for ex in batch_examples]
                    # input_hindi = torch.tensor(fst_hindi.get_phrase_vector(batch_english))
                    # output_dict = model.predict(input1=input_english, input2=input_hindi, targets=targets)
                    batch_english = [
                        " ".join([token for token, tag in zip(ex.text_pp.split(" "), ex.langids_pp.split(" "))])
                        for ex in batch_examples]
                    input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
                    output_dict = model.predict(input1=input_english, targets=targets)
                elif args.model_name == "charlstmlstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_idxs, batch_char_lengths, batch_lengths, targets=targets)
                elif args.model_name == "sclstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_screps, batch_lengths = sc_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_screps, batch_lengths, targets=targets)
                elif args.model_name == "fasttext-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_embs, batch_lengths = fst_english.get_pad_vectors(
                        batch_tokens=[line.split(" ") for line in batch_sentences],
                        return_lengths=True)
                    output_dict = model.predict(batch_embs, batch_lengths, targets=targets)
                batch_loss = output_dict["loss"].cpu().detach().numpy()
                dev_loss += batch_loss
                dev_acc += output_dict["acc_num"]
                dev_preds.extend(output_dict["preds"])
                dev_true.extend(targets)
                # update progress
                progress_bar(batch_id + 1, n_batches,
                             ["batch_time", "batch_loss", "avg_batch_loss", "batch_acc", 'avg_batch_acc'],
                             [time.time() - st_time, batch_loss, dev_loss / (batch_id + 1),
                              output_dict["acc_num"] / dev_batch_size, dev_acc / ((batch_id + 1) * dev_batch_size)])
                # break
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
                    torch.save({
                        'epoch_id': epoch_id,
                        'max_dev_acc': best_dev_acc,
                        'argmax_dev_acc': best_dev_acc_epoch,
                        'model_state_dict': model.state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict()
                    },
                        os.path.join(CHECKPOINT_PATH, name))
                    printlog("Model saved at {} in epoch {}".format(os.path.join(CHECKPOINT_PATH, name), epoch_id))
                    if label_vocab is not None:
                        json.dump(label_vocab._asdict(), open(os.path.join(CHECKPOINT_PATH, "label_vocab.json"), "w"),
                                  indent=4)
                        printlog("label_vocab saved at {} in epoch {}".format(
                            os.path.join(CHECKPOINT_PATH, "label_vocab.json"), epoch_id))
                    if word_vocab is not None:
                        json.dump(word_vocab._asdict(), open(os.path.join(CHECKPOINT_PATH, "word_vocab.json"), "w"),
                                  indent=4)
                        printlog("word_vocab saved at {} in epoch {}".format(
                            os.path.join(CHECKPOINT_PATH, "word_vocab.json"), epoch_id))
                        opfile = open(os.path.join(CHECKPOINT_PATH, "vocab.txt"), "w")
                        for word in word_vocab.token2idx.keys():
                            opfile.write(word + "\n")
                        opfile.close()
                        printlog(
                            "vocab words saved at {} in epoch {}".format(os.path.join(CHECKPOINT_PATH, "vocab.txt"),
                                                                         epoch_id))
                        opfile = open(os.path.join(CHECKPOINT_PATH, "vocab_char.txt"), "w")
                        for word in word_vocab.chartoken2idx.keys():
                            opfile.write(word + "\n")
                        opfile.close()
                        printlog("vocab chars saved at {} in epoch {}".format(
                            os.path.join(CHECKPOINT_PATH, "vocab_char.txt"), epoch_id))
                    if lid_label_vocab is not None:
                        json.dump(lid_label_vocab._asdict(),
                                  open(os.path.join(CHECKPOINT_PATH, "lid_label_vocab.json"), "w"), indent=4)
                        printlog("lid_label_vocab saved at {} in epoch {}".format(
                            os.path.join(CHECKPOINT_PATH, "lid_label_vocab.json"), epoch_id))
                    if pos_label_vocab is not None:
                        json.dump(pos_label_vocab._asdict(),
                                  open(os.path.join(CHECKPOINT_PATH, "pos_label_vocab.json"), "w"), indent=4)
                        printlog("pos_label_vocab saved at {} in epoch {}".format(
                            os.path.join(CHECKPOINT_PATH, "pos_label_vocab.json"), epoch_id))
                else:
                    printlog("no improvements in results to save a checkpoint")
                    printlog(f"checkpoint previously saved during epoch {best_dev_acc_epoch}(0-base) at: "
                             f"{os.path.join(CHECKPOINT_PATH, name)}")
            else:
                if (start_epoch == 0 and epoch_id == start_epoch) or best_dev_f1 < dev_f1:
                    best_dev_f1, best_dev_f1_epoch = dev_f1, epoch_id
                    torch.save({
                        'epoch_id': epoch_id,
                        'max_dev_f1': best_dev_f1,
                        'argmax_dev_f1': best_dev_f1_epoch,
                        'model_state_dict': model.state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict()
                    },
                        os.path.join(CHECKPOINT_PATH, name))
                    printlog("Model saved at {} in epoch {}".format(os.path.join(CHECKPOINT_PATH, name), epoch_id))
                    if label_vocab is not None:
                        json.dump(label_vocab._asdict(), open(os.path.join(CHECKPOINT_PATH, "label_vocab.json"), "w"),
                                  indent=4)
                        printlog("label_vocab saved at {} in epoch {}".format(
                            os.path.join(CHECKPOINT_PATH, "label_vocab.json"), epoch_id))
                    if word_vocab is not None:
                        json.dump(word_vocab._asdict(), open(os.path.join(CHECKPOINT_PATH, "word_vocab.json"), "w"),
                                  indent=4)
                        printlog("word_vocab saved at {} in epoch {}".format(
                            os.path.join(CHECKPOINT_PATH, "word_vocab.json"), epoch_id))
                        opfile = open(os.path.join(CHECKPOINT_PATH, "vocab.txt"), "w")
                        for word in word_vocab.token2idx.keys():
                            opfile.write(word + "\n")
                        opfile.close()
                        printlog(
                            "vocab words saved at {} in epoch {}".format(os.path.join(CHECKPOINT_PATH, "vocab.txt"),
                                                                         epoch_id))
                        opfile = open(os.path.join(CHECKPOINT_PATH, "vocab_char.txt"), "w")
                        for word in word_vocab.chartoken2idx.keys():
                            opfile.write(word + "\n")
                        opfile.close()
                        printlog("vocab chars saved at {} in epoch {}".format(
                            os.path.join(CHECKPOINT_PATH, "vocab_char.txt"), epoch_id))
                    if lid_label_vocab is not None:
                        json.dump(lid_label_vocab._asdict(),
                                  open(os.path.join(CHECKPOINT_PATH, "lid_label_vocab.json"), "w"), indent=4)
                        printlog("lid_label_vocab saved at {} in epoch {}".format(
                            os.path.join(CHECKPOINT_PATH, "lid_label_vocab.json"), epoch_id))
                    if pos_label_vocab is not None:
                        json.dump(pos_label_vocab._asdict(),
                                  open(os.path.join(CHECKPOINT_PATH, "pos_label_vocab.json"), "w"), indent=4)
                        printlog("pos_label_vocab saved at {} in epoch {}".format(
                            os.path.join(CHECKPOINT_PATH, "pos_label_vocab.json"), epoch_id))
                else:
                    printlog("no improvements in results to save a checkpoint")
                    printlog(f"checkpoint previously saved during epoch {best_dev_f1_epoch}(0-base) at: "
                             f"{os.path.join(CHECKPOINT_PATH, name)}")

        # if "test" in args.mode:
        #     args.mode = "test"

    """ testing """
    for selected_examples in [dev_examples, test_examples]:
        if selected_examples:

            """ testing on dev and test set """
            printlog("\n\n################")
            printlog(f"in testing...loading model.pth.tar from {CHECKPOINT_PATH}")
            model.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, "model.pth.tar"),
                                             map_location=torch.device(DEVICE))['model_state_dict'])
            save_errors_path = os.path.join(CHECKPOINT_PATH, str(datetime.datetime.now()).replace(" ", "_"))
            test_exs, test_preds, test_probs, test_true, targets = [], [], [], [], None
            n_batches = int(np.ceil(len(selected_examples) / dev_batch_size))
            selected_examples_batch_iter = batch_iter(selected_examples, dev_batch_size, shuffle=False)
            printlog(f"len of {args.mode} data: {len(selected_examples)}")
            printlog(f"n_batches of {args.mode} data: {n_batches}")
            for batch_id, batch_examples in enumerate(selected_examples_batch_iter):
                st_time = time.time()
                # forward
                targets = [label_vocab.token2idx[ex.label] if ex.label is not None else ex.label for ex in
                           batch_examples]
                targets = None if any([x is None for x in targets]) else targets
                if args.sentence_transformers:
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "bert-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "bert-sc-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_screps, _ = sc_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                                batch_screps=batch_screps, targets=targets)
                elif args.model_name in ["bert-charlstm-lstm", "bert-charlstm-lstm-v2"]:
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                                batch_idxs=batch_idxs, batch_char_lengths=batch_char_lengths,
                                                batch_lengths=batch_lengths, targets=targets)
                elif args.model_name == "bert-fasttext-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_embs, batch_lengths = fst_english.get_pad_vectors(
                        batch_tokens=[line.split(" ") for line in batch_sentences],
                        return_lengths=True)
                    output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                                batch_screps=batch_embs, targets=targets)
                elif "bert" in args.model_name and args.model_name.startswith("li-"):
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_lang_ids = [getattr(ex, f"langids_{args.text_type}" if args.text_type != "" else "langids")
                                      for ex in batch_examples]
                    # adding "other" at ends
                    batch_lang_ids = [" ".join([str(lid_label_vocab.sos_token_idx)] +
                                               [str(lid_label_vocab.token2idx[lang]) for lang in lang_ids.split(" ")] +
                                               [str(lid_label_vocab.eos_token_idx)])
                                      for lang_ids in batch_lang_ids]
                    output_dict = model.predict(batch_sentences, batch_lang_ids, targets=targets)
                elif "bert" in args.model_name and args.model_name.startswith("posi-"):
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_pos_ids = [getattr(ex, f"postags_{args.text_type}" if args.text_type != "" else "postags")
                                     for ex in batch_examples]
                    # adding "other" at ends
                    batch_pos_ids = [" ".join([str(pos_label_vocab.sos_token_idx)] +
                                              [str(pos_label_vocab.token2idx[lang]) for lang in pos_ids.split(" ")] +
                                              [str(pos_label_vocab.eos_token_idx)])
                                     for pos_ids in batch_pos_ids]
                    output_dict = model.predict(batch_sentences, batch_pos_ids, targets=targets)
                elif args.model_name == "bert-semantic-similarity":
                    batch_sentences = []
                    for text_type in ["src", "tgt"]:
                        batch_sentences.extend([getattr(ex, text_type) for ex in batch_examples])
                    output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "cnn-character-bert":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif "bert" in args.model_name:
                    if args.fusion_text_types:
                        batch_sentences = []
                        for text_type in args.fusion_text_types:
                            batch_sentences.extend([getattr(ex, text_type) for ex in batch_examples])
                        output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                    elif args.multitask_lid_sa:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        lid_targets = [[lid_label_vocab.token2idx[token] for token in getattr(ex,
                                                                                              f"langids_{args.langids_type}" if args.langids_type != "" else "langids").split(
                            " ")]
                                       for ex in batch_examples]
                        output_dict = model.predict(text_batch=batch_sentences, sa_targets=targets,
                                                    lid_targets=lid_targets)
                    elif args.sentence_bert:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                    else:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "fasttext-vanilla":
                    # batch_english = [" ".join([token for token, tag in zip(ex.text_trt.split(" "), ex.langids_pp.split(" "))
                    #                            if tag.lower() != "hin"]) for ex in batch_examples]
                    # input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
                    # batch_hindi = [" ".join([token for token, tag in zip(ex.text_trt.split(" "), ex.langids_pp.split(" "))
                    #                          if tag.lower() == "hin"]) for ex in batch_examples]
                    # input_hindi = torch.tensor(fst_hindi.get_phrase_vector(batch_english))
                    # output_dict = model.predict(input1=input_english, input2=input_hindi, targets=targets)
                    batch_english = [
                        " ".join([token for token, tag in zip(ex.text_pp.split(" "), ex.langids_pp.split(" "))])
                        for ex in batch_examples]
                    input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
                    output_dict = model.predict(input1=input_english, targets=targets)
                elif args.model_name == "charlstmlstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_idxs, batch_char_lengths, batch_lengths, targets=targets)
                elif args.model_name == "sclstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_screps, batch_lengths = sc_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_screps, batch_lengths, targets=targets)
                elif args.model_name == "fasttext-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_embs, batch_lengths = fst_english.get_pad_vectors(
                        batch_tokens=[line.split(" ") for line in batch_sentences],
                        return_lengths=True)
                    output_dict = model.predict(batch_embs, batch_lengths, targets=targets)
                test_exs.extend(batch_examples)
                test_preds.extend(output_dict["preds"])
                test_probs.extend(output_dict["probs"])
                if targets is not None:
                    test_true.extend(targets)
                # update progress
                progress_bar(batch_id + 1, n_batches, ["batch_time"], [time.time() - st_time])
            printlog("")

            printlog(f"\n(NEW!) saving predictions in the folder: {save_errors_path}")
            create_path(save_errors_path)
            opfile = jsonlines.open(os.path.join(save_errors_path, "predictions.jsonl"), "w")
            for i, (x, y, z) in enumerate(zip(test_exs, test_preds, test_probs)):
                dt = x._asdict()
                dt.update({"prediction": label_vocab.idx2token[y]})
                dt.update({"pred_probs": z})
                opfile.write(dt)
            opfile.close()
            opfile = open(os.path.join(save_errors_path, "predictions.txt"), "w")
            for i, (x, y) in enumerate(zip(test_exs, test_preds)):
                try:
                    opfile.write(f"{label_vocab.idx2token[y]} ||| {x.text}\n")
                except AttributeError:
                    opfile.write(f"{label_vocab.idx2token[y]}\n")
            opfile.close()

            if targets is not None and len(targets) > 0:
                printlog(f"\n(NEW!) saving errors files in the folder: {save_errors_path}")
                # report
                report = classification_report(test_true, test_preds, digits=4,
                                               target_names=[label_vocab.idx2token[idx]
                                                             for idx in range(0, label_vocab.n_all_tokens)])
                printlog("\n" + report)
                opfile = open(os.path.join(save_errors_path, "report.txt"), "w")
                opfile.write(report + "\n")
                opfile.close()
                # errors
                opfile = jsonlines.open(os.path.join(save_errors_path, "errors.jsonl"), "w")
                for i, (x, y, z) in enumerate(zip(test_exs, test_preds, test_true)):
                    if y != z:
                        dt = x._asdict()
                        dt.update({"prediction": label_vocab.idx2token[y]})
                        opfile.write(dt)
                opfile.close()
                for idx_i in label_vocab.idx2token:
                    for idx_j in label_vocab.idx2token:
                        opfile = jsonlines.open(os.path.join(save_errors_path,
                                                             f"errors_pred-{idx_i}_target-{idx_j}.jsonl"), "w")
                        temp_test_exs = [x for x, y, z in zip(test_exs, test_preds, test_true)
                                         if (y == idx_i and z == idx_j)]
                        for x in temp_test_exs:
                            dt = x._asdict()
                            dt.update({"prediction": label_vocab.idx2token[idx_i]})
                            opfile.write(dt)
                        opfile.close()
                # confusion matrix
                cm = confusion_matrix(y_true=test_true, y_pred=test_preds, labels=list(set(test_true)))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                              display_labels=[label_vocab.idx2token[ii] for ii in list(set(test_true))])
                disp = disp.plot(values_format="d")
                get_module_or_attr("matplotlib", "pyplot").savefig(
                    os.path.join(save_errors_path, "confusion_matrix.png"))
                # plt.show()

    """ interactive """
    # if args.mode == "interactive":
    #     printlog("\n\n################")
    #     printlog(f"in interactive inference mode...loading model.pth.tar from {CHECKPOINT_PATH}")
    #     model.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, "model.pth.tar"),
    #                                      map_location=torch.device(DEVICE))['model_state_dict'])
    #     while True:
    #         text_input = input("enter your text here: ")
    #         if text_input == "-1":
    #             break
    #         new_example = EXAMPLE(dataset=None, task=None, split_type=None, uid=None, text=text_input, text_pp=None,
    #                               label=None, langids=None, seq_labels=None, langids_pp=None, meta_data=None)
    #         test_examples = [new_example]
    #
    #         # -----------> left unedited from previous
    #         test_exs, test_preds, test_probs, test_true, targets = [], [], [], [], None
    #         selected_examples = test_examples
    #         n_batches = int(np.ceil(len(selected_examples) / dev_batch_size))
    #         selected_examples_batch_iter = batch_iter(selected_examples, dev_batch_size, shuffle=False)
    #         printlog(f"len of {args.mode} data: {len(selected_examples)}")
    #         printlog(f"n_batches of {args.mode} data: {n_batches}")
    #         for batch_id, batch_examples in enumerate(selected_examples_batch_iter):
    #             st_time = time.time()
    #             # forward
    #             targets = [label_vocab.token2idx[ex.label] if ex.label is not None else ex.label for ex in
    #                        batch_examples]
    #             targets = None if any([x is None for x in targets]) else targets
    #             if args.model_name == "bert-lstm":
    #                 batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
    #                                    for ex in batch_examples]
    #                 output_dict = model.predict(text_batch=batch_sentences, targets=targets)
    #             elif args.model_name == "bert-sc-lstm":
    #                 batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
    #                                    for ex in batch_examples]
    #                 batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(
    #                     batch_sentences, model.bert_tokenizer, max_len=200)
    #                 batch_screps, _ = sc_tokenize(batch_sentences, word_vocab)
    #                 output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
    #                                             batch_screps=batch_screps, targets=targets)
    #             elif args.model_name in ["bert-charlstm-lstm", "bert-charlstm-lstm-v2"]:
    #                 batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
    #                                    for ex in batch_examples]
    #                 batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(
    #                     batch_sentences, model.bert_tokenizer, max_len=200)
    #                 batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
    #                 output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
    #                                             batch_idxs=batch_idxs, batch_char_lengths=batch_char_lengths,
    #                                             batch_lengths=batch_lengths, targets=targets)
    #             elif args.model_name == "bert-fasttext-lstm":
    #                 batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
    #                                    for ex in batch_examples]
    #                 batch_sentences, batch_bert_dict, batch_splits = bert_subword_tokenize(
    #                     batch_sentences, model.bert_tokenizer, max_len=200)
    #                 batch_embs, batch_lengths = fst_english.get_pad_vectors(
    #                     batch_tokens=[line.split(" ") for line in batch_sentences],
    #                     return_lengths=True)
    #                 output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
    #                                             batch_screps=batch_embs, targets=targets)
    #             elif "bert" in args.model_name and args.model_name.startswith("li-"):
    #                 batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
    #                                    for ex in batch_examples]
    #                 batch_lang_ids = [getattr(ex, f"langids_{args.text_type}" if args.text_type != "" else "langids")
    #                                   for ex in batch_examples]
    #                 # adding "other" at ends
    #                 batch_lang_ids = [" ".join([str(lid_label_vocab.sos_token_idx)] +
    #                                            [str(lid_label_vocab.token2idx[lang]) for lang in lang_ids.split(" ")] +
    #                                            [str(lid_label_vocab.eos_token_idx)])
    #                                   for lang_ids in batch_lang_ids]
    #                 output_dict = model.predict(batch_sentences, batch_lang_ids, targets=targets)
    #             elif "bert" in args.model_name and args.model_name.startswith("posi-"):
    #                 batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
    #                                    for ex in batch_examples]
    #                 batch_pos_ids = [getattr(ex, f"postags_{args.text_type}" if args.text_type != "" else "postags")
    #                                  for ex in batch_examples]
    #                 # adding "other" at ends
    #                 batch_pos_ids = [" ".join([str(pos_label_vocab.sos_token_idx)] +
    #                                           [str(pos_label_vocab.token2idx[lang]) for lang in pos_ids.split(" ")] +
    #                                           [str(pos_label_vocab.eos_token_idx)])
    #                                  for pos_ids in batch_pos_ids]
    #                 output_dict = model.predict(batch_sentences, batch_pos_ids, targets=targets)
    #             elif args.model_name == "bert-semantic-similarity":
    #                 batch_sentences = []
    #                 for text_type in ["src", "tgt"]:
    #                     batch_sentences.extend([getattr(ex, text_type) for ex in batch_examples])
    #                 output_dict = model.predict(text_batch=batch_sentences, targets=targets)
    #             elif "bert" in args.model_name:
    #                 if args.fusion_text_types:
    #                     batch_sentences = []
    #                     for text_type in args.fusion_text_types:
    #                         batch_sentences.extend([getattr(ex, text_type) for ex in batch_examples])
    #                     output_dict = model.predict(text_batch=batch_sentences, targets=targets)
    #                 elif args.multitask_lid_sa:
    #                     batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
    #                                        for ex in batch_examples]
    #                     lid_targets = [[lid_label_vocab.token2idx[token] for token in
    #                                     getattr(ex,
    #                                             f"langids_{args.langids_type}" if args.langids_type != "" else "langids").split(
    #                                         " ")]
    #                                    for ex in batch_examples]
    #                     output_dict = model.predict(text_batch=batch_sentences, sa_targets=targets,
    #                                                 lid_targets=lid_targets)
    #                 elif args.sentence_bert:
    #                     batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
    #                                        for ex in batch_examples]
    #                     output_dict = model.predict(text_batch=batch_sentences, targets=targets)
    #                 else:
    #                     batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
    #                                        for ex in batch_examples]
    #                     output_dict = model.predict(text_batch=batch_sentences, targets=targets)
    #             elif args.model_name == "fasttext-vanilla":
    #                 # batch_english = [" ".join([token for token, tag in zip(ex.text_trt.split(" "), ex.langids_pp.split(" "))
    #                 #                            if tag.lower() != "hin"]) for ex in batch_examples]
    #                 # input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
    #                 # batch_hindi = [" ".join([token for token, tag in zip(ex.text_trt.split(" "), ex.langids_pp.split(" "))
    #                 #                          if tag.lower() == "hin"]) for ex in batch_examples]
    #                 # input_hindi = torch.tensor(fst_hindi.get_phrase_vector(batch_english))
    #                 # output_dict = model.predict(input1=input_english, input2=input_hindi, targets=targets)
    #                 batch_english = [
    #                     " ".join([token for token, tag in zip(ex.text_pp.split(" "), ex.langids_pp.split(" "))])
    #                     for ex in batch_examples]
    #                 input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
    #                 output_dict = model.predict(input1=input_english, targets=targets)
    #             elif args.model_name == "charlstmlstm":
    #                 batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
    #                                    for ex in batch_examples]
    #                 batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
    #                 output_dict = model.predict(batch_idxs, batch_char_lengths, batch_lengths, targets=targets)
    #             elif args.model_name == "sclstm":
    #                 batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
    #                                    for ex in batch_examples]
    #                 batch_screps, batch_lengths = sc_tokenize(batch_sentences, word_vocab)
    #                 output_dict = model.predict(batch_screps, batch_lengths, targets=targets)
    #             elif args.model_name == "fasttext-lstm":
    #                 batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
    #                                    for ex in batch_examples]
    #                 batch_embs, batch_lengths = fst_english.get_pad_vectors(
    #                     batch_tokens=[line.split(" ") for line in batch_sentences],
    #                     return_lengths=True)
    #                 output_dict = model.predict(batch_embs, batch_lengths, targets=targets)
    #             test_exs.extend(batch_examples)
    #             test_preds.extend(output_dict["preds"])
    #             test_probs.extend(output_dict["probs"])
    #             if targets is not None:
    #                 test_true.extend(targets)
    #             # update progress
    #             # progress_bar(batch_id + 1, n_batches, ["batch_time"], [time.time() - st_time])
    #         # <-----------
    #         printlog([label_vocab.idx2token[y] for y in test_preds])

    if logger_file_name and not args.debug:
        os.system(f"cat {logger_file_name}")


def run_adaptation(model_name_or_path: str, data_folder: str, save_folder=None):
    path_name, model_name = os.path.split(model_name_or_path)
    save_folder = save_folder or os.path.join(data_folder, f"models/{model_name}")
    if os.path.exists(save_folder):
        raise Exception(f"save_folder: `{save_folder}` already exists. Aborting adaptation training.")
    else:
        print(f"Saving checkpoints at: {save_folder}")

    formatted_script = \
        f"{'CUDA_VISIBLE_DEVICES=0 python' if torch.cuda.is_available() else 'python'} " \
        f"{os.path.join(SRC_ROOT_PATH, 'misc', 'run_bert_mlm.py')} " \
        f"--overwrite_output_dir --logging_steps 500 --save_steps 5000 --save_total_limit 1 --num_train_epochs 3.0 " \
        f"--output_dir={save_folder} --model_name_or_path={model_name_or_path} " \
        f"--do_train --train_data_file={os.path.join(data_folder, 'train.txt')} " \
        f"--line_by_line --mlm " \
        f"--per_device_train_batch_size=8 --per_device_eval_batch_size=8 " \
        f"--do_eval --eval_data_file={os.path.join(data_folder, 'test.txt')}"
    print(formatted_script)

    os.system(formatted_script)
    return
