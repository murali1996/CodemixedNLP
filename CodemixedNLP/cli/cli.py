import os
import time
from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
import torch

from ..benchmarks.helpers import get_model_nparams, progress_bar, batch_iter
from ..benchmarks.models import WholeWordBertMLP, WholeWordBertForSeqClassificationAndTagging
from ..benchmarks.helpers import load_vocab
from ..datasets import EXAMPLE
from ..paths import ARXIV_CHECKPOINTS

from ..utils import is_module_available, get_module_or_attr


class ModelCreator:

    @staticmethod
    def validate_name(name):
        choices = ["sentiment", "aggression", "hatespeech", "lid", "pos", "ner", "mt"]
        assert name in choices, print(f"possible names for models: {choices}")
        return

    def from_pretrained(self, name):
        self.validate_name(name)
        if name in ["sentiment", "aggression", "hatespeech"]:
            classifier = ClassificationModel.from_pretrained(name)  # automatically identifies downloaded path
            classifier.load()
            return classifier
        elif name in ["lid", "pos", "ner"]:
            tagger = TaggerModel.from_pretrained(name)  # automatically identifies downloaded path
            tagger.load()
            return tagger
        elif name in ["mt"]:
            if not is_module_available("fairseq"):
                raise ImportError("For loading MT model, Fairseq library must be installed. "
                                  "See `Installation-Extras` in `README.md` for more details")
            translator = MachineTranslationModel.from_pretrained(name)
            translator.load()
            return translator
        else:
            raise NotImplementedError

    @staticmethod
    def show_available_tasks():
        tasks = ["sentiment", "aggression", "hatespeech", "lid", "pos", "ner", "mt"]
        print(tasks)
        return


class ModelBase(ABC):

    def __init__(self, pretrained_name_or_path):
        self.pretrained_name_or_path = pretrained_name_or_path
        self.ready = False
        self.vocabs = {}
        self.model = None
        print(self)

    @abstractmethod
    def load_required_vocab(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def get_predictions(self, input_sentences: Union[str, List[str]], batch_size=8, verbose=False):
        raise NotImplementedError


class ClassificationModel(ModelBase):

    def __init__(self, pretrained_name_or_path=None, **kwargs):
        super(ClassificationModel, self).__init__(pretrained_name_or_path)
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    def set_device(self, device: str):
        assert device in ["cuda", "cpu"]
        self.device = device

    def validate_ckpt_path(self, ckpt_path):
        ckpt_path = ckpt_path or self.pretrained_name_or_path
        assert ckpt_path and os.path.exists(ckpt_path), print(ckpt_path)
        return ckpt_path

    def load_required_vocab(self, ckpt_path=None):
        ckpt_path = self.validate_ckpt_path(ckpt_path)
        vocab_names = ["label_vocab", "lid_label_vocab", "pos_label_vocab", "word_vocab"]
        vocabs = {nm: None for nm in vocab_names}
        something_exists = False

        for nm in vocab_names:
            pth = os.path.join(ckpt_path, f"{nm}.json")
            if os.path.exists(pth):
                vocabs[nm] = load_vocab(pth)
                something_exists = True

        if not something_exists:
            raise Exception(f"no vocab files exist in the path specified: {ckpt_path}")

        self.vocabs = vocabs

        return

    def load(self, ckpt_path=None, device=None, vocabs=None):
        ckpt_path = self.validate_ckpt_path(ckpt_path)
        device = device or self.device
        vocabs = vocabs or self.vocabs
        label_vocab = vocabs.get("label_vocab", None)
        if not label_vocab:
            self.load_required_vocab(ckpt_path)
            label_vocab = self.vocabs.get("label_vocab", None)
        assert label_vocab, print(f"Couldn't find `label_vocab` in vocabs. Found: {vocabs}. "
                                  f"Consider running `load_required_vocab` with correct path for loading vocab.")
        model = WholeWordBertMLP(out_dim=label_vocab.n_all_tokens, pretrained_path="xlm-roberta-base",
                                 finetune_bert=False)
        model.to(device)
        print(f"number of parameters (all, trainable) in your model: {get_model_nparams(model)}")
        print(f"in interactive inference mode...loading model.pth.tar from {ckpt_path}")
        model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.pth.tar"),
                                         map_location=torch.device(device))['model_state_dict'])

        self.model = model
        self.ready = True
        return

    def get_predictions(self, input_sentences: Union[str, List[str]], batch_size=8, verbose=False) -> List[str]:

        if isinstance(input_sentences, str):
            input_sentences = [input_sentences, ]

        test_examples = []
        for sent in input_sentences:
            sent = sent.strip()
            new_example = EXAMPLE(dataset=None, task=None, split_type=None, uid=None, text=sent, text_pp=None,
                                  label=None, langids=None, seq_labels=None, langids_pp=None, meta_data=None)
            test_examples.append(new_example)

        label_vocab = self.vocabs.get("label_vocab", None)
        test_exs, test_preds, test_probs, test_true, targets = [], [], [], [], None
        selected_examples = test_examples
        n_batches = int(np.ceil(len(selected_examples) / batch_size))
        selected_examples_batch_iter = batch_iter(selected_examples, batch_size, shuffle=False)
        print(f"len of data: {len(selected_examples)}")
        print(f"n_batches of data: {n_batches}")
        for batch_id, batch_examples in enumerate(selected_examples_batch_iter):
            st_time = time.time()
            # forward
            targets = [label_vocab.token2idx[ex.label] if ex.label is not None else ex.label for ex in batch_examples]
            targets = None if any([x is None for x in targets]) else targets

            batch_sentences = [getattr(ex, "text") for ex in batch_examples]
            output_dict = self.model.predict(text_batch=batch_sentences, targets=targets)

            test_exs.extend(batch_examples)
            test_preds.extend(output_dict["preds"])
            test_probs.extend(output_dict["probs"])
            if targets is not None:
                test_true.extend(targets)
            # update progress
            if verbose:
                progress_bar(batch_id + 1, n_batches, ["batch_time"], [time.time() - st_time])

        results = [label_vocab.idx2token[y] for y in test_preds]
        if verbose:
            print(results)

        return results

    @classmethod
    def from_pretrained(cls, name: str):
        assert name in ["sentiment", "aggression", "hatespeech"]
        return cls(ARXIV_CHECKPOINTS[name])

    def __repr__(self):
        return f"ClassificationModel:: pretrained_name_or_path: {self.pretrained_name_or_path}, ready: {self.ready}"


class TaggerModel(ModelBase):

    def __init__(self, pretrained_name_or_path=None, **kwargs):
        super(TaggerModel, self).__init__(pretrained_name_or_path)
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    def set_device(self, device: str):
        assert device in ["cuda", "cpu"]
        self.device = device

    def validate_ckpt_path(self, ckpt_path):
        ckpt_path = ckpt_path or self.pretrained_name_or_path
        assert ckpt_path and os.path.exists(ckpt_path), print(ckpt_path)
        return ckpt_path

    def load_required_vocab(self, ckpt_path=None):
        ckpt_path = self.validate_ckpt_path(ckpt_path)
        vocab_names = ["label_vocab", "lid_label_vocab", "pos_label_vocab", "word_vocab"]
        vocabs = {nm: None for nm in vocab_names}
        something_exists = False

        for nm in vocab_names:
            pth = os.path.join(ckpt_path, f"{nm}.json")
            if os.path.exists(pth):
                vocabs[nm] = load_vocab(pth)
                something_exists = True

        if not something_exists:
            raise Exception(f"no vocab files exist in the path specified: {ckpt_path}")

        self.vocabs = vocabs

        return

    def load(self, ckpt_path=None, device=None, vocabs=None):
        ckpt_path = self.validate_ckpt_path(ckpt_path)
        device = device or self.device
        vocabs = vocabs or self.vocabs
        label_vocab = vocabs.get("label_vocab", None)
        if not label_vocab:
            self.load_required_vocab(ckpt_path)
            label_vocab = self.vocabs.get("label_vocab", None)
        assert label_vocab, print(f"Couldn't find `label_vocab` in vocabs. Found: {vocabs}. "
                                  f"Consider running `load_required_vocab` with correct path for loading vocab.")
        model = WholeWordBertForSeqClassificationAndTagging(
            sent_out_dim=2,  # Any random number because we don't care about classification loss
            lang_out_dim=label_vocab.n_tokens,
            pretrained_path="xlm-roberta-base")
        model.to(device)
        print(f"number of parameters (all, trainable) in your model: {get_model_nparams(model)}")
        print(f"in interactive inference mode...loading model.pth.tar from {ckpt_path}")
        model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.pth.tar"),
                                         map_location=torch.device(device))['model_state_dict'])

        self.model = model
        self.ready = True
        return

    def get_predictions(self, input_sentences: Union[str, List[str]], batch_size=1, verbose=False, pretify=False) -> \
            List[str]:

        assert batch_size == 1, \
            print("this constraint enforced due to the way the predict_lid() method is written")

        if isinstance(input_sentences, str):
            input_sentences = [input_sentences, ]

        test_examples = []
        for sent in input_sentences:
            sent = sent.strip()
            new_example = EXAMPLE(dataset=None, task=None, split_type=None, uid=None, text=sent, text_pp=None,
                                  label=None, langids=None, seq_labels=None, langids_pp=None, meta_data=None)
            test_examples.append(new_example)

        label_vocab = self.vocabs["label_vocab"]
        test_exs, results = [], []
        selected_examples = test_examples
        n_batches = int(np.ceil(len(selected_examples) / batch_size))
        selected_examples_batch_iter = batch_iter(selected_examples, batch_size, shuffle=False)
        print(f"len of data: {len(selected_examples)}")
        print(f"n_batches of data: {n_batches}")
        for batch_id, batch_examples in enumerate(selected_examples_batch_iter):
            st_time = time.time()

            batch_sentences = [getattr(ex, "text") for ex in batch_examples]
            output_dict = self.model.predict_lid(text_batch=batch_sentences)

            test_exs.extend(batch_examples)

            # update progress
            if verbose:
                progress_bar(batch_id + 1, n_batches, ["batch_time"], [time.time() - st_time])

            tags_ = [label_vocab.idx2token[y] for y in output_dict["preds"]]

            if pretify:
                results_ = " ".join(
                    ["\\".join([str(i) for i in itm]) for itm in list(zip(batch_sentences[0].split(), tags_))])
            else:
                results_ = " ".join(tags_)

            if verbose:
                print(results_)
            results.append(results_)

        return results

    @classmethod
    def from_pretrained(cls, name: str):
        assert name in ["lid", "pos", "ner"]
        return cls(ARXIV_CHECKPOINTS[name])

    def __repr__(self):
        return f"TaggerModel:: pretrained_name_or_path: {self.pretrained_name_or_path}, ready: {self.ready}"


class MachineTranslationModel(ModelBase):

    def __init__(self, pretrained_name_or_path=None, **kwargs):
        super(MachineTranslationModel, self).__init__(pretrained_name_or_path)
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        print(self.pretrained_name_or_path)

    def set_device(self, device: str):
        assert device in ["cuda", "cpu"]
        self.device = device

    def validate_ckpt_path(self, ckpt_path):
        ckpt_path = ckpt_path or self.pretrained_name_or_path
        assert ckpt_path and os.path.exists(ckpt_path), print(ckpt_path)
        return ckpt_path

    def load_required_vocab(self, ckpt_path=None):
        return

    def load(self, ckpt_path=None, device=None, vocabs=None):
        ckpt_path = self.validate_ckpt_path(ckpt_path)
        device = device or self.device
        TransformerModel = get_module_or_attr("fairseq.models.transformer", "TransformerModel")
        model = TransformerModel.from_pretrained(
            ckpt_path,
            "checkpoint_best.pt",
            tokenizer="moses",
            bpe="sentencepiece",
            sentencepiece_model=os.path.join(ckpt_path, "spm8000.model"),
            max_sentences=2,
        )
        model.to(device)
        print(f"number of parameters (all, trainable) in your model: {get_model_nparams(model)}")
        print(f"in interactive inference mode...loading model.pth.tar from {ckpt_path}")
        self.model = model
        self.ready = True
        return

    def get_predictions(self, input_sentences: Union[str, List[str]], batch_size=8, verbose=False) -> List[str]:

        if isinstance(input_sentences, str):
            input_sentences = [input_sentences, ]

        results = self.model.translate(input_sentences, beam=5)

        if verbose:
            print(results)

        return results

    @classmethod
    def from_pretrained(cls, name: str):
        assert name in ["mt"]
        print(ARXIV_CHECKPOINTS[name])
        return cls(ARXIV_CHECKPOINTS[name])

    def __repr__(self):
        return f"MachineTranslationModel:: pretrained_name_or_path: {self.pretrained_name_or_path}, ready: {self.ready}"
