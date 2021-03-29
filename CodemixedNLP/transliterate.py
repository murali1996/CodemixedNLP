import copy
import os

import jsonlines
from tqdm import tqdm

from .utils import get_module_or_attr, is_module_available

_INDICTRANS_TRANSLITERATORS = None
if is_module_available("indictrans"):
    _INDICTRANS_TRANSLITERATORS = {
        "eng_to_hin": get_module_or_attr("indictrans", "Transliterator")(source="eng", target="hin"),
        "hin_to_eng": get_module_or_attr("indictrans", "Transliterator")(source="hin", target="eng")
    }


def transliterate_indictrans(files, text_type, src_lang="eng", tgt_lang="hin", langids_type=None,
                             src_folder=None, dest_folder=None):
    """
    A simple word-by-word transliterator based on indictrans

    :param files: a list of files to load and add a transliteration (can be abssolute or relative paths)
    :param text_type: the key to be used in each jsonline as input text stream for transliteration
    :param src_lang: the script type of the input text stream, eg. `eng` for latin
    :param tgt_lang: the script type of the target text stream, eg. `hin` for devanagari
    :param langids_type: if avaialble and is not None, used to segregate/add non-Hindi and non-English parts to jsonline
    :param src_folder: if not None, file names can be relative to this path
    :param dest_folder: if not None, the augmented jsonlines are saved in this folder with same file name
    """

    if not is_module_available("indictrans"):
        raise ImportError("Install `indictrans` by following install-extras in the docs")

    text_type_tail = "_D" if tgt_lang == "hin" else "_E"

    if not dest_folder and not src_folder:
        raise Exception("one of `dest_folder` or `src_folder` need to be a valid path")

    if src_folder:
        assert os.path.exists(src_folder)
        files = [os.path.join(src_folder, file) for file in files]

    if dest_folder:
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
    else:
        dest_folder = os.path.split(src_folder)[0]

    for path in files:
        print(f"reading from {path}")
        samples = [line for line in jsonlines.open(path, "r")]
        new_samples = []
        for sample in tqdm(samples):
            new_sample = copy.deepcopy(sample)
            tokens = sample[text_type].split(" ")
            src2tgt = _INDICTRANS_TRANSLITERATORS[f"{src_lang}_to_{tgt_lang}"]
            new_tokens = [src2tgt.transform(token) for token in tokens]
            new_sample[text_type + text_type_tail] = " ".join(new_tokens)
            if langids_type and langids_type in sample and sample[langids_type]:
                langids = sample[langids_type].split(" ")
                assert len(langids) == len(tokens) == len(new_tokens)
                non_english = [token for token, langid in zip(tokens, langids) if langid != "en"]
                non_hindi = [token for token, langid in zip(tokens, langids) if langid != "hi"]
                non_english_devanagari = [token for token, langid in zip(new_tokens, langids) if langid != "en"]
                new_sample[text_type + "_non_english"] = " ".join(non_english)
                new_sample[text_type + "_non_hindi"] = " ".join(non_hindi)
                new_sample[text_type + "_non_english_D"] = " ".join(non_english_devanagari)
            new_samples.append(new_sample)
        with jsonlines.open(os.path.join(dest_folder, os.path.split(path)[-1]), 'w') as writer:
            for new_sample in new_samples:
                writer.write(new_sample)

    return
