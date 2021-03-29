import io
import json
import sys
from collections import namedtuple
from typing import List, Union
import json
import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from ..paths import CHECKPOINTS_PATH
from ..utils import get_module_or_attr

""" ##################### """
"""       helpers         """
""" ##################### """


def get_model_nparams(model):
    """
    :param model: can be a list of nn.Module or a single nn.Module
    :return: (all, trainable parameters)
    """
    if not isinstance(model, list):
        items = [model, ]
    else:
        items = model
    ntotal, n_gradrequired = 0, 0
    for item in items:
        for param in list(item.parameters()):
            temp = 1
            for sz in list(param.size()):
                temp *= sz
            ntotal += temp
            if param.requires_grad:
                n_gradrequired += temp
    return ntotal, n_gradrequired


def train_validation_split(data, train_ratio, seed=11927):
    len_ = len(data)
    train_len_ = int(np.ceil(train_ratio * len_))
    inds_shuffled = np.arange(len_)
    np.random.seed(seed)
    np.random.shuffle(inds_shuffled)
    train_data = []
    for ind in inds_shuffled[:train_len_]:
        train_data.append(data[ind])
    validation_data = []
    for ind in inds_shuffled[train_len_:]:
        validation_data.append(data[ind])
    return train_data, validation_data


def batch_iter(data, batch_size, shuffle):
    """
    each data item is a tuple of labels and text
    """
    n_batches = int(np.ceil(len(data) / batch_size))
    indices = list(range(len(data)))
    if shuffle:
        np.random.shuffle(indices)

    for i in range(n_batches):
        batch_indices = indices[i * batch_size: (i + 1) * batch_size]
        yield [data[i] for i in batch_indices]


def progress_bar(value, endvalue, names=[], values=[], bar_length=15):
    assert (len(names) == len(values))
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    string = ''
    for name, val in zip(names, values):
        temp = '|| {0}: {1:.4f} '.format(name, val) if val else '|| {0}: {1} '.format(name, None)
        string += temp
    sys.stdout.write("\rPercent: [{0}] {1}% {2}".format(arrow + spaces, int(round(percent * 100)), string))
    sys.stdout.flush()
    if value >= endvalue - 1:
        print()
    return


class FastTextVecs(object):
    def __init__(self, langauge, dim=300, path=None):

        path = path or f'{CHECKPOINTS_PATH}/fasttext_models/cc.{langauge}.{dim}.bin'
        if path.endswith(".bin"):  # a model instead of word vectors
            self.ft = get_module_or_attr("fasttext").load_model(path)
            self.word_vectors = None
            self.words = None
            self.ft_dim = self.ft.get_dimension()
        elif path.endswith(".vec"):
            self.ft = None
            self.word_vectors = self.load_vectors(path)
            self.words = [*self.word_vectors.keys()]
            self.ft_dim = len(self.word_vectors[self.words[0]])
        else:
            raise Exception(f"Invalid extension for the FASTTEXT_MODEL_PATH: {path}")
        print(f'fasttext model loaded from: {path} with dim: {self.ft_dim}')

    def get_dimension(self):
        return self.ft_dim

    def get_word_vector(self, word):
        if self.ft is not None:
            return self.ft.get_word_vector(word)
        try:
            word_vector = self.word_vectors[word]
        except KeyError:
            word_vector = np.array([0.0] * self.ft_dim)
        return word_vector

    def get_phrase_vector(self, phrases: List[str]):
        if isinstance(phrases, str):
            phrases = [phrases]
        assert isinstance(phrases, list) or isinstance(phrases, tuple), print(type(phrases))
        batch_array = np.array([np.mean([self.get_word_vector(word) for word in sentence.split()], axis=0)
                                if sentence else self.get_word_vector("")
                                for sentence in phrases])
        return batch_array

    def get_pad_vectors(self, batch_tokens: Union[List[str], List[List[str]]], token_pad_idx=0.0, return_lengths=False):
        if isinstance(batch_tokens[0], list):
            tensors_list = [torch.tensor([self.get_word_vector(token) for token in line]) for line in batch_tokens]
        elif isinstance(batch_tokens[0], str):
            tensors_list = \
                [torch.tensor([self.get_word_vector(token) for token in line.split(" ")]) for line in batch_tokens]
        else:
            raise ValueError
        batch_vectors = pad_sequence(tensors_list, batch_first=True, padding_value=token_pad_idx)
        if return_lengths:
            batch_lengths = torch.tensor([len(x) for x in tensors_list]).long()
            return batch_vectors, batch_lengths
        return batch_vectors

    @staticmethod
    def load_vectors(fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array([*map(float, tokens[1:])])
        return data


""" ##################### """
"""    vocab functions    """
""" ##################### """


def load_vocab(path) -> namedtuple:
    return_dict = json.load(open(path))
    # for idx2token, idx2chartoken, have to change keys from strings to ints
    #   https://stackoverflow.com/questions/45068797/how-to-convert-string-int-json-into-real-int-with-json-loads
    if "token2idx" in return_dict:
        return_dict.update({"idx2token": {v: k for k, v in return_dict["token2idx"].items()}})
    if "chartoken2idx" in return_dict:
        return_dict.update({"idx2chartoken": {v: k for k, v in return_dict["chartoken2idx"].items()}})

    # NEW
    # vocab: dict to named tuple
    vocab = namedtuple('vocab', sorted(return_dict))
    return vocab(**return_dict)


def create_vocab(data: List[str],
                 keep_simple=False,
                 min_max_freq: tuple = (1, float("inf")),
                 topk=None,
                 intersect: List = None,
                 load_char_tokens: bool = False,
                 is_label: bool = False,
                 labels_data_split_at_whitespace: bool = False) -> namedtuple:
    """
    :param data: list of sentences from which tokens are obtained as whitespace seperated
    :param keep_simple: retain tokens that have ascii and do not have digits (for preprocessing)
    :param min_max_freq: retain tokens whose count satisfies >min_freq and <max_freq
    :param topk: retain only topk tokens (specify either topk or min_max_freq)
    :param intersect: retain tokens that are at intersection with a custom token list
    :param load_char_tokens: if true, character tokens will also be loaded
    :param is_label: when the inouts are list of labels
    :param labels_data_split_at_whitespace:
    :return: a vocab namedtuple
    """

    if topk is None and (min_max_freq[0] > 1 or min_max_freq[1] < float("inf")):
        raise Exception("both min_max_freq and topk should not be provided at once !")

    # if is_label
    if is_label:

        def split_(txt: str):
            if labels_data_split_at_whitespace:
                return txt.split(" ")
            else:
                return [txt, ]

        # get all tokens
        token_freq, token2idx, idx2token = {}, {}, {}
        for example in tqdm(data):
            for token in split_(example):
                if token not in token_freq:
                    token_freq[token] = 0
                token_freq[token] += 1
        print(f"Total tokens found: {len(token_freq)}")
        print(f"token_freq:\n{token_freq}\n")

        # create token2idx and idx2token
        for token in token_freq:
            idx = len(token2idx)
            idx2token[idx] = token
            token2idx[token] = idx

        token_freq = list(sorted(token_freq.items(), key=lambda item: item[1], reverse=True))
        return_dict = {"token2idx": token2idx,
                       "idx2token": idx2token,
                       "token_freq": token_freq,
                       "n_tokens": len(token2idx),
                       "n_all_tokens": len(token2idx),
                       "pad_token_idx": -1}

    else:

        # get all tokens
        token_freq, token2idx, idx2token = {}, {}, {}
        for example in tqdm(data):
            for token in example.split(" "):
                if token not in token_freq:
                    token_freq[token] = 0
                token_freq[token] += 1
        print(f"Total tokens found: {len(token_freq)}")

        # retain only simple tokens
        if keep_simple:
            isascii = lambda s: len(s) == len(s.encode())
            hasdigits = lambda s: len([x for x in list(s) if x.isdigit()]) > 0
            tf = [(t, f) for t, f in [*token_freq.items()] if (isascii(t) and not hasdigits(t))]
            token_freq = {t: f for (t, f) in tf}
            print(f"After removing non-ascii and tokens with digits, total tokens retained: {len(token_freq)}")

        # retain only tokens with specified min and max range
        if min_max_freq[0] > 1 or min_max_freq[1] < float("inf"):
            sorted_ = sorted(token_freq.items(), key=lambda item: item[1], reverse=True)
            tf = [(i[0], i[1]) for i in sorted_ if (min_max_freq[0] <= i[1] <= min_max_freq[1])]
            token_freq = {t: f for (t, f) in tf}
            print(f"After min_max_freq selection, total tokens retained: {len(token_freq)}")

        # retain only topk tokens
        if topk is not None:
            sorted_ = sorted(token_freq.items(), key=lambda item: item[1], reverse=True)
            token_freq = {t: f for (t, f) in list(sorted_)[:topk]}
            print(f"After topk selection, total tokens retained: {len(token_freq)}")

        # retain only interection of tokens
        if intersect is not None and len(intersect) > 0:
            tf = [(t, f) for t, f in [*token_freq.items()] if (t in intersect or t.lower() in intersect)]
            token_freq = {t: f for (t, f) in tf}
            print(f"After intersection, total tokens retained: {len(token_freq)}")

        # create token2idx and idx2token
        for token in token_freq:
            idx = len(token2idx)
            idx2token[idx] = token
            token2idx[token] = idx

        # add <<PAD>> special token
        ntokens = len(token2idx)
        pad_token = "<<PAD>>"
        token_freq.update({pad_token: -1})
        token2idx.update({pad_token: ntokens})
        idx2token.update({ntokens: pad_token})

        # add <<UNK>> special token
        ntokens = len(token2idx)
        unk_token = "<<UNK>>"
        token_freq.update({unk_token: -1})
        token2idx.update({unk_token: ntokens})
        idx2token.update({ntokens: unk_token})

        # new
        # add <<EOS>> special token
        ntokens = len(token2idx)
        eos_token = "<<EOS>>"
        token_freq.update({eos_token: -1})
        token2idx.update({eos_token: ntokens})
        idx2token.update({ntokens: eos_token})

        # new
        # add <<SOS>> special token
        ntokens = len(token2idx)
        sos_token = "<<SOS>>"
        token_freq.update({sos_token: -1})
        token2idx.update({sos_token: ntokens})
        idx2token.update({ntokens: sos_token})

        # return dict
        token_freq = list(sorted(token_freq.items(), key=lambda item: item[1], reverse=True))
        return_dict = {"token2idx": token2idx,
                       "idx2token": idx2token,
                       "token_freq": token_freq,
                       "pad_token": pad_token,
                       "pad_token_idx": token2idx[pad_token],
                       "unk_token": unk_token,
                       "unk_token_idx": token2idx[unk_token],
                       "eos_token": eos_token,
                       "eos_token_idx": token2idx[eos_token],
                       "sos_token": sos_token,
                       "sos_token_idx": token2idx[sos_token],
                       "n_tokens": len(token2idx) - 4,
                       "n_special_tokens": 4,
                       "n_all_tokens": len(token2idx)
                       }

        # load_char_tokens
        if load_char_tokens:
            print("loading character tokens as well")
            char_return_dict = create_char_vocab(use_default=True, data=data)
            return_dict.update(char_return_dict)

    # NEW
    # vocab: dict to named tuple
    vocab = namedtuple('vocab', sorted(return_dict))
    return vocab(**return_dict)


def create_char_vocab(use_default: bool, data=None) -> dict:
    if not use_default and data is None:
        raise Exception("data is None")

    # reset char token utils
    chartoken2idx, idx2chartoken = {}, {}
    char_unk_token, char_pad_token, char_start_token, char_end_token = \
        "<<CHAR_UNK>>", "<<CHAR_PAD>>", "<<CHAR_START>>", "<<CHAR_END>>"
    special_tokens = [char_unk_token, char_pad_token, char_start_token, char_end_token]
    for char in special_tokens:
        idx = len(chartoken2idx)
        chartoken2idx[char] = idx
        idx2chartoken[idx] = char

    if use_default:
        chars = list(
            """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
        for char in chars:
            if char not in chartoken2idx:
                idx = len(chartoken2idx)
                chartoken2idx[char] = idx
                idx2chartoken[idx] = char
    else:
        # helper funcs
        # isascii = lambda s: len(s) == len(s.encode())
        """
        # load batches of lines and obtain unique chars
        nlines = len(data)
        bsize = 5000
        nbatches = int( np.ceil(nlines/bsize) )
        for i in tqdm(range(nbatches)):
            blines = " ".join( [ex for ex in data[i*bsize:(i+1)*bsize]] )
            #bchars = set(list(blines))
            for char in bchars:
                if char not in chartoken2idx:
                    idx = len(chartoken2idx)
                    chartoken2idx[char] = idx
                    idx2chartoken[idx] = char
        """
        # realized the method above doesn't preserve order!!
        for line in tqdm(data):
            for char in line:
                if char not in chartoken2idx:
                    idx = len(chartoken2idx)
                    chartoken2idx[char] = idx
                    idx2chartoken[idx] = char

    print(f"number of unique chars found: {len(chartoken2idx)}")
    return_dict = {"chartoken2idx": chartoken2idx,
                   "idx2chartoken": idx2chartoken,
                   "char_unk_token": char_unk_token,
                   "char_pad_token": char_pad_token,
                   "char_start_token": char_start_token,
                   "char_end_token": char_end_token,
                   "char_unk_token_idx": chartoken2idx[char_unk_token],
                   "char_pad_token_idx": chartoken2idx[char_pad_token],
                   "char_start_token_idx": chartoken2idx[char_start_token],
                   "char_end_token_idx": chartoken2idx[char_end_token],
                   "n_tokens": len(chartoken2idx) - 4,
                   "n_special_tokens": 4}
    return return_dict


""" ##################### """
"""     tokenizers        """
""" ##################### """


class Tokenizer:
    def __init__(self,
                 word_vocab=None,
                 tag_input_label_vocab=None,
                 bert_tokenizer=None):

        self.word_vocab = word_vocab
        self.tag_input_label_vocab = tag_input_label_vocab
        self.bert_tokenizer = bert_tokenizer
        self.fastTextVecs = None

        # self.tokenize = None  # assign to right tokenize func from below based on the requirement

    def load_tag_vocab(self, data):
        self.tag_input_label_vocab = create_vocab(data, is_label=False, labels_data_split_at_whitespace=True)

    def save_tag_vocab_to_checkpoint(self, ckpt_dir):
        if not self.tag_input_label_vocab:
            print("`tag_input_label_vocab` is None and need to be loaded first")
            return

        json.dump(self.tag_input_label_vocab._asdict(),
                  open(os.path.join(ckpt_dir, "tag_input_label_vocab.json"), "w"),
                  indent=4)
        return

    def load_tag_vocab_from_checkpoint(self, ckpt_dir):
        assert self.tag_input_label_vocab is None, print("`tag_input_label_vocab` is not None and overwriting it")
        self.tag_input_label_vocab = load_vocab(os.path.join(ckpt_dir, "tag_input_label_vocab.json"))
        return

    def load_word_vocab(self, data):
        self.word_vocab = create_vocab(data, is_label=False, load_char_tokens=True)

    def save_word_vocab_to_checkpoint(self, ckpt_dir):
        if not self.word_vocab:
            print("`word_vocab` is None and need to be loaded first")
            return

        json.dump(self.word_vocab._asdict(),
                  open(os.path.join(ckpt_dir, "word_vocab.json"), "w"),
                  indent=4)
        return

    def load_word_vocab_from_checkpoint(self, ckpt_dir):
        assert self.word_vocab is None, print("`word_vocab` is not None and overwriting it")
        self.word_vocab = load_vocab(os.path.join(ckpt_dir, "word_vocab.json"))
        return

    def bert_subword_tokenize(self,
                              batch_sentences,
                              bert_tokenizer=None,
                              batch_tag_sequences=None,
                              max_len=512,
                              as_dict=True):

        bert_tokenizer = bert_tokenizer or self.bert_tokenizer
        text_padding_idx = bert_tokenizer.pad_token_id

        if batch_tag_sequences is not None:
            assert self.tag_input_label_vocab, \
                print(f"`tag_input_label_vocab` is required for processing batch_tag_sequences")

        if batch_tag_sequences is not None:
            assert len(batch_tag_sequences) == len(batch_sentences)
            # adding "other" at ends, and converting them to idxs
            batch_tag_sequences = (
                [" ".join([str(self.tag_input_label_vocab.sos_token_idx)] +
                          [str(self.tag_input_label_vocab.token2idx[tag]) for tag in tag_sequence.split(" ")] +
                          [str(self.tag_input_label_vocab.eos_token_idx)])
                 for tag_sequence in batch_tag_sequences]
            )
            trimmed_batch_sentences = [
                _custom_bert_tokenize_sentence_with_lang_ids(text, bert_tokenizer, max_len, tag_ids)
                for text, tag_ids in zip(batch_sentences, batch_tag_sequences)]
            batch_sentences, batch_tokens, batch_splits, batch_tag_ids = list(zip(*trimmed_batch_sentences))
            batch_encoded_dicts = [bert_tokenizer.encode_plus(tokens) for tokens in batch_tokens]
            batch_input_ids = pad_sequence(
                [torch.tensor(encoded_dict["input_ids"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
                padding_value=text_padding_idx)
            batch_attention_masks = pad_sequence(
                [torch.tensor(encoded_dict["attention_mask"]) for encoded_dict in batch_encoded_dicts],
                batch_first=True,
                padding_value=0)
            batch_token_type_ids = pad_sequence(
                [torch.tensor([int(sidx) for sidx in tag_ids.split(" ")]) for tag_ids in batch_tag_ids],
                batch_first=True,
                padding_value=self.tag_input_label_vocab.pad_token_idx)
            batch_bert_dict = {
                "attention_mask": batch_attention_masks,
                "input_ids": batch_input_ids,
                "token_type_ids": batch_token_type_ids
            }
        else:
            # batch_sentences = [text if text.strip() else "." for text in batch_sentences]
            trimmed_batch_sentences = [_custom_bert_tokenize_sentence(text, bert_tokenizer, max_len) for text in
                                       batch_sentences]
            batch_sentences, batch_tokens, batch_splits = list(zip(*trimmed_batch_sentences))
            batch_encoded_dicts = [bert_tokenizer.encode_plus(tokens) for tokens in batch_tokens]
            batch_input_ids = pad_sequence(
                [torch.tensor(encoded_dict["input_ids"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
                padding_value=text_padding_idx)
            batch_attention_masks = pad_sequence(
                [torch.tensor(encoded_dict["attention_mask"]) for encoded_dict in batch_encoded_dicts],
                batch_first=True,
                padding_value=0)
            # batch_token_type_ids = pad_sequence(
            #     [torch.tensor(encoded_dict["token_type_ids"]) for encoded_dict in batch_encoded_dicts],
            #     batch_first=True,
            #     padding_value=0)
            batch_bert_dict = {
                "attention_mask": batch_attention_masks,
                "input_ids": batch_input_ids
            }

        batch_lengths = [len(sent.split(" ")) for sent in batch_sentences]  # useful for lstm based downstream layers

        if as_dict:
            return {
                "features": batch_bert_dict,
                "batch_splits": batch_splits,
                "batch_sentences": batch_sentences,
                "batch_lengths": batch_lengths,  # new
                "batch_char_lengths": None,  # new
                "batch_size": len(batch_sentences),  # new
            }

        # not returning `batch_lengths` for backward compatability
        return batch_sentences, batch_bert_dict, batch_splits

    def word_tokenize(self,
                      batch_sentences,
                      vocab=None,
                      as_dict=True):
        raise NotImplementedError

    def fasttext_tokenize(self,
                          batch_sentences,
                          vocab=None,
                          as_dict=True):
        if not self.fastTextVecs:
            self.fastTextVecs = FastTextVecs(langauge="en")

        batch_vectors, batch_lengths = self.fastTextVecs.get_pad_vectors(batch_sentences, return_lengths=True)

        if as_dict:
            return {
                "features": {
                    "batch_tensor": batch_vectors,
                    "batch_lengths": batch_lengths
                },
                "batch_splits": None,
                "batch_sentences": None,
                "batch_lengths": batch_lengths,
                "batch_size": len(batch_sentences)
            }

        return batch_vectors, batch_lengths

    def char_tokenize(self,
                      batch_sentences,
                      vocab=None,
                      as_dict=True):
        """
        :returns List[pad_sequence], Tensor[int]
        """

        vocab = vocab or self.word_vocab

        chartoken2idx = vocab.chartoken2idx
        char_unk_token = vocab.char_unk_token
        char_pad_token = vocab.char_pad_token
        char_start_token = vocab.char_start_token
        char_end_token = vocab.char_end_token

        def func_word2charids(word):
            return [chartoken2idx[char_start_token]] + \
                   [chartoken2idx[char] if char in chartoken2idx else chartoken2idx[char_unk_token]
                    for char in list(word)] + \
                   [chartoken2idx[char_end_token]]

        char_idxs = [[func_word2charids(word) for word in sent.split(" ")] for sent in batch_sentences]
        char_padding_idx = chartoken2idx[char_pad_token]
        batch_idxs = [pad_sequence(
            [torch.as_tensor(list_of_wordidxs).long() for list_of_wordidxs in list_of_lists],
            batch_first=True,
            padding_value=char_padding_idx
        )
            for list_of_lists in char_idxs]
        # dim [nsentences,nwords_per_sentence]
        nchars = [torch.as_tensor([len(wordlevel) for wordlevel in sentlevel]).long() for sentlevel in char_idxs]
        # dim [nsentences]
        nwords = torch.tensor([len(sentlevel) for sentlevel in batch_idxs]).long()

        if as_dict:
            return {
                "features": {
                    "batch_tensor": batch_idxs,
                    "batch_lengths": nwords,
                    "batch_char_lengths": nchars
                },
                "batch_splits": None,
                "batch_sentences": None,
                "batch_lengths": nwords,
                "batch_size": len(batch_sentences)
            }

        return batch_idxs, nchars, nwords

    def sc_tokenize(self,
                    batch_sentences,
                    vocab=None,
                    as_dict=True):
        """
        return List[pad_sequence], Tensor[int]
        """

        vocab = vocab or self.word_vocab

        chartoken2idx = vocab.chartoken2idx
        char_unk_token_idx = vocab.char_unk_token_idx

        def sc_vector(word):
            a = [0] * len(chartoken2idx)
            if word[0] in chartoken2idx:
                a[chartoken2idx[word[0]]] = 1
            else:
                a[char_unk_token_idx] = 1
            b = [0] * len(chartoken2idx)
            for char in word[1:-1]:
                if char in chartoken2idx: b[chartoken2idx[char]] += 1
                # else: b[ char_unk_token_idx ] = 1
            c = [0] * len(chartoken2idx)
            if word[-1] in chartoken2idx:
                c[chartoken2idx[word[-1]]] = 1
            else:
                c[char_unk_token_idx] = 1
            return a + b + c

        # return list of tesnors and we don't need to pad these unlike cnn-lstm case!
        batch_encoding = pad_sequence([torch.tensor([sc_vector(word) for word in sent.split(" ")]).float() for sent in
                                       batch_sentences], batch_first=True)
        nwords = torch.tensor([len(sentlevel) for sentlevel in batch_encoding]).long()

        if as_dict:
            return {
                "features": {
                    "batch_tensor": batch_encoding,
                    "batch_lengths": nwords,
                    "batch_char_lengths": None,
                },
                "batch_splits": None,
                "batch_sentences": None,
                "batch_lengths": nwords,
                "batch_size": len(batch_sentences)
            }

        return batch_encoding, nwords


def word_tokenize(batch_sentences, vocab, as_dict=False):
    raise NotImplementedError


def fasttext_tokenize(batch_sentences, vocab=None, as_dict=False):
    raise NotImplementedError


def char_tokenize(batch_sentences, vocab, as_dict=False):
    """
    :returns List[pad_sequence], Tensor[int]
    """
    chartoken2idx = vocab.chartoken2idx
    char_unk_token = vocab.char_unk_token
    char_pad_token = vocab.char_pad_token
    char_start_token = vocab.char_start_token
    char_end_token = vocab.char_end_token

    def func_word2charids(word):
        return [chartoken2idx[char_start_token]] + \
               [chartoken2idx[char] if char in chartoken2idx else chartoken2idx[char_unk_token]
                for char in list(word)] + \
               [chartoken2idx[char_end_token]]

    char_idxs = [[func_word2charids(word) for word in sent.split(" ")] for sent in batch_sentences]
    char_padding_idx = chartoken2idx[char_pad_token]
    batch_idxs = [pad_sequence(
        [torch.as_tensor(list_of_wordidxs).long() for list_of_wordidxs in list_of_lists],
        batch_first=True,
        padding_value=char_padding_idx
    )
        for list_of_lists in char_idxs]
    # dim [nsentences,nwords_per_sentence]
    nchars = [torch.as_tensor([len(wordlevel) for wordlevel in sentlevel]).long() for sentlevel in char_idxs]
    # dim [nsentences]
    nwords = torch.tensor([len(sentlevel) for sentlevel in batch_idxs]).long()

    if as_dict:
        return {
            "features": {
                "batch_tensor": batch_idxs,
                "batch_lengths": nwords,
                "batch_char_lengths": nchars
            },
            "batch_splits": None,
            "batch_sentences": None,
            "batch_lengths": nwords,
            "batch_size": len(batch_sentences)
        }

    return batch_idxs, nchars, nwords


def sc_tokenize(batch_sentences, vocab, as_dict=False):
    """
    return List[pad_sequence], Tensor[int]
    """
    chartoken2idx = vocab.chartoken2idx
    char_unk_token_idx = vocab.char_unk_token_idx

    def sc_vector(word):
        a = [0] * len(chartoken2idx)
        if word[0] in chartoken2idx:
            a[chartoken2idx[word[0]]] = 1
        else:
            a[char_unk_token_idx] = 1
        b = [0] * len(chartoken2idx)
        for char in word[1:-1]:
            if char in chartoken2idx: b[chartoken2idx[char]] += 1
            # else: b[ char_unk_token_idx ] = 1
        c = [0] * len(chartoken2idx)
        if word[-1] in chartoken2idx:
            c[chartoken2idx[word[-1]]] = 1
        else:
            c[char_unk_token_idx] = 1
        return a + b + c

    # return list of tesnors and we don't need to pad these unlike cnn-lstm case!
    batch_encoding = [torch.tensor([sc_vector(word) for word in sent.split(" ")]).float() for sent in batch_sentences]
    nwords = torch.tensor([len(sentlevel) for sentlevel in batch_encoding]).long()

    if as_dict:
        return {
            "features": {
                "batch_tensor": batch_encoding,
                "batch_lengths": nwords,
                "batch_char_lengths": None,
            },
            "batch_splits": None,
            "batch_sentences": None,
            "batch_lengths": nwords,
            "batch_size": len(batch_sentences)
        }

    return batch_encoding, nwords


def bert_subword_tokenize(batch_sentences,
                          bert_tokenizer,
                          max_len=512,
                          batch_lang_sequences=None,
                          text_padding_idx=None,
                          token_type_padding_idx=None,
                          as_dict=False):
    """
    inputs:
        batch_sentences: List[str]
            a list of textual sentences to tokenized
        bert_tokenizer: transformers.BertTokenizer
            a valid tokenizer that can tokenize into sub-words starting with "#"
        max_len: Int
            maximum length to chunk/pad input text
        batch_lang_sequences: List[str]
            each `str` in the list corrresponds to a space (i.e " ") seperated sequence of language tags
            that are already converted to idx, so that these are easily modified to create `batch_token_type_ids`
    outputs:
        batch_attention_masks, batch_input_ids, batch_token_type_ids
            2d tensors of shape (bs,max_len)
        batch_splits: List[List[Int]]
            specifies number of sub-tokens for each word in each sentence after sub-word bert tokenization
    """

    text_padding_idx = text_padding_idx or bert_tokenizer.pad_token_id
    if batch_lang_sequences is not None:
        token_type_padding_idx = token_type_padding_idx or 0

    if batch_lang_sequences is not None:
        assert len(batch_lang_sequences) == len(batch_sentences)
        trimmed_batch_sentences = [_custom_bert_tokenize_sentence_with_lang_ids(text, bert_tokenizer, max_len, lang_ids)
                                   for text, lang_ids in zip(batch_sentences, batch_lang_sequences)]
        batch_sentences, batch_tokens, batch_splits, batch_lang_ids = list(zip(*trimmed_batch_sentences))
        batch_encoded_dicts = [bert_tokenizer.encode_plus(tokens) for tokens in batch_tokens]
        batch_input_ids = pad_sequence(
            [torch.tensor(encoded_dict["input_ids"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
            padding_value=text_padding_idx)
        batch_attention_masks = pad_sequence(
            [torch.tensor(encoded_dict["attention_mask"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
            padding_value=0)
        # batch_token_type_ids = pad_sequence(
        #     [torch.tensor(encoded_dict["token_type_ids"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
        #     padding_value=0)
        batch_token_type_ids = pad_sequence(
            [torch.tensor([int(sidx) for sidx in lang_ids.split(" ")]) for lang_ids in batch_lang_ids],
            batch_first=True,
            padding_value=token_type_padding_idx)
        batch_bert_dict = {
            "attention_mask": batch_attention_masks,
            "input_ids": batch_input_ids,
            "token_type_ids": batch_token_type_ids
        }
    else:
        # batch_sentences = [text if text.strip() else "." for text in batch_sentences]
        trimmed_batch_sentences = [_custom_bert_tokenize_sentence(text, bert_tokenizer, max_len) for text in
                                   batch_sentences]
        batch_sentences, batch_tokens, batch_splits = list(zip(*trimmed_batch_sentences))
        batch_encoded_dicts = [bert_tokenizer.encode_plus(tokens) for tokens in batch_tokens]
        batch_input_ids = pad_sequence(
            [torch.tensor(encoded_dict["input_ids"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
            padding_value=text_padding_idx)
        batch_attention_masks = pad_sequence(
            [torch.tensor(encoded_dict["attention_mask"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
            padding_value=0)
        batch_bert_dict = {
            "attention_mask": batch_attention_masks,
            "input_ids": batch_input_ids
        }

    batch_lengths = [len(sent.split(" ")) for sent in batch_sentences]  # useful for lstm based downstream layers

    if as_dict:
        return {
            "features": batch_bert_dict,
            "batch_splits": batch_splits,
            "batch_sentences": batch_sentences,
            "batch_lengths": batch_lengths,  # new
            "batch_char_lengths": None,  # new
            "batch_size": len(batch_sentences),  # new
        }

    # not returning `batch_lengths` for backward compatability
    return batch_sentences, batch_bert_dict, batch_splits


def _tokenize_untokenize(input_text: str, bert_tokenizer):
    subtokens = bert_tokenizer.tokenize(input_text)
    output = []
    for subt in subtokens:
        if subt.startswith("##"):
            output[-1] += subt[2:]
        else:
            output.append(subt)
    return " ".join(output)


def _custom_bert_tokenize_sentence(input_text, bert_tokenizer, max_len):
    tokens = []
    split_sizes = []
    text = []
    # for token in _tokenize_untokenize(input_text, bert_tokenizer).split(" "):
    for token in input_text.split(" "):
        word_tokens = bert_tokenizer.tokenize(token)
        if len(tokens) + len(word_tokens) > max_len - 2:  # 512-2 = 510
            break
        if len(word_tokens) == 0:
            continue
        tokens.extend(word_tokens)
        split_sizes.append(len(word_tokens))
        text.append(token)

    return " ".join(text), tokens, split_sizes


def _custom_bert_tokenize_sentence_with_lang_ids(input_text, bert_tokenizer, max_len, input_lang_ids):
    tokens = []
    split_sizes = []
    text = []
    lang_ids = []

    # the 2 is substracted due to added terminal start/end positions
    assert len(input_text.split(" ")) == len(input_lang_ids.split(" ")) - 2, \
        print(len(input_text.split(" ")), len(input_lang_ids.split(" ")) - 2)

    lids = input_lang_ids.split(" ")
    non_terminal_lids = lids[1:-1]

    # cannot use _tokenize_untokenize(input_text) because doing so might change the one-one mapping between
    #   input_text and non_terminal_lids
    for token, lid in zip(input_text.split(" "), non_terminal_lids):
        word_tokens = bert_tokenizer.tokenize(token)
        if len(tokens) + len(word_tokens) > max_len - 2:  # 512-2 = 510
            break
        if len(word_tokens) == 0:
            continue
        tokens.extend(word_tokens)
        split_sizes.append(len(word_tokens))
        text.append(token)
        lang_ids.extend([lid] * len(word_tokens))
    lang_ids = [lids[0]] + lang_ids + [lids[-1]]

    return " ".join(text), tokens, split_sizes, " ".join(lang_ids)


def merge_subword_encodings_for_words(bert_seq_encodings,
                                      seq_splits,
                                      mode='avg',
                                      keep_terminals=False,
                                      device=torch.device("cpu")):
    bert_seq_encodings = bert_seq_encodings[:sum(seq_splits) + 2, :]  # 2 for [CLS] and [SEP]
    bert_cls_enc = bert_seq_encodings[0:1, :]
    bert_sep_enc = bert_seq_encodings[-1:, :]
    bert_seq_encodings = bert_seq_encodings[1:-1, :]
    # a tuple of tensors
    split_encoding = torch.split(bert_seq_encodings, seq_splits, dim=0)
    batched_encodings = pad_sequence(split_encoding, batch_first=True, padding_value=0)
    if mode == 'avg':
        seq_splits = torch.tensor(seq_splits).reshape(-1, 1).to(device)
        out = torch.div(torch.sum(batched_encodings, dim=1), seq_splits)
    elif mode == "add":
        out = torch.sum(batched_encodings, dim=1)
    elif mode == "first":
        out = batched_encodings[:, 0, :]
    else:
        raise Exception("Not Implemented")

    if keep_terminals:
        out = torch.cat((bert_cls_enc, out, bert_sep_enc), dim=0)
    return out


def merge_subword_encodings_for_sentences(bert_seq_encodings,
                                          seq_splits):
    bert_seq_encodings = bert_seq_encodings[:sum(seq_splits) + 2, :]  # 2 for [CLS] and [SEP]
    bert_cls_enc = bert_seq_encodings[0:1, :]
    bert_sep_enc = bert_seq_encodings[-1:, :]
    bert_seq_encodings = bert_seq_encodings[1:-1, :]
    return torch.mean(bert_seq_encodings, dim=0)
