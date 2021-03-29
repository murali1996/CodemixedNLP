import csv
import os
import re
from collections import namedtuple
from time import time
from typing import List

import jsonlines
from tqdm import tqdm

from ._preprocess import clean_generic, clean_sail2017_lines, clean_sentimix2020_lines
from .benchmarks.helpers import progress_bar

# from indictrans import Transliterator
# TRANSLITERATOR = Transliterator(source='eng', target='hin')

# Make sure export GOOGLE_APPLICATION_CREDENTIALS=My\ First\ Project-62b86dcd7bce.json
# from google.cloud import translate

MAX_CHAR_LEN = None
SEED = 11927
FIELDS = ["dataset", "task", "split_type", "uid",
          "text", "langids", "label", "seq_labels",
          "text_pp", "langids_pp",
          "meta_data"]
EXAMPLE = namedtuple(f"example", FIELDS)  # namedtuple(f"example", FIELDS, defaults=(None,) * len(FIELDS))


class ExampleClass:
    def __init__(self, name, fields=None, defaults=None):
        self.name = name
        if fields:
            if not defaults:
                defaults = [None] * len(fields)
            assert len(defaults) == len(fields)
            for f, d in zip(fields, defaults):
                setattr(self, f, d)

    def __call__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def __repr__(self):
        return f"<EXAMPLE: {self.name}>"


# Make sure export GOOGLE_APPLICATION_CREDENTIALS=My\ First\ Project-62b86dcd7bce.json
def google_translate(text_list, target_language):
    # print(text)
    client = translate.TranslationServiceClient()
    parent = "projects/hybrid-formula-290820/locations/global"
    response_list = []

    response = client.translate_text(text_list, target_language, parent)
    for translation in response.translations:
        response_list.append(translation.translated_text)
    # print(response.translations[0].translated_text)
    return response_list


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


def read_csv_file(path, has_header=True, delimiter=","):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        if has_header:
            lines = [row for row in csv_reader][1:]
        else:
            lines = [row for row in csv_reader]
    return lines


def read_jsonl_file(path):
    return [line for line in jsonlines.open(path, "r")]


def read_datasets_jsonl(path, mode=""):
    examples = []
    for i, line in enumerate(jsonlines.open(path)):
        if i == 0:
            fields_ = line.keys()
            Example = namedtuple(f"{mode}_example", fields_)
            # Example = namedtuple(f"{mode}_example", fields_, defaults=(None,) * len(fields_))
        examples.append(Example(**line))
    print(f"in read_datasets_jsonl(): path:{path}, mode:{mode}, #examples:{len(examples)}")
    return examples


def read_datasets_jsonl_new(path, mode=""):
    examples = []
    fields_ = []
    for i, line in enumerate(jsonlines.open(path)):
        if i == 0:
            fields_ = line.keys()
        example = ExampleClass(f"{mode}_example", fields_)
        examples.append(example(**line))
    print(f"in read_datasets_jsonl(): path:{path}, mode:{mode}, #examples:{len(examples)}")
    return examples


def read_lince_downloads(path, mode, dataset_name, task_name, is_pos=False, is_ner=False, standardizing_tags={}):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN
    tokens, langids, postags, nertags, label, uid = [], [], [], [], None, None
    FIELDS += [fieldname for fieldname in ("nertags", "postags") if fieldname not in FIELDS]

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    all_lines = open(path, "r").readlines()
    for line_num, line in enumerate(all_lines):
        if line.strip() == "":
            if mode == "test":
                uid = len(examples)
            txt = " ".join(tokens)
            new_txt = clean_generic(txt)
            if new_txt.strip() == "":
                new_txt = txt
            if len(new_txt) > MAX_CHAR_LEN:
                n_trimmed += 1
                newtokens, currsum = [], 0
                for tkn in new_txt.split():  # 1 for space
                    if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                        newtokens.append(tkn)
                        currsum += len(tkn) + 1
                    else:
                        break
                new_txt = " ".join(newtokens)
            example = Example(dataset=dataset_name,
                              task=task_name,
                              split_type=mode,
                              uid=uid,
                              label=label,
                              text=txt,
                              text_pp=new_txt,
                              langids=" ".join([standardizing_tags[lid] if lid in standardizing_tags else "other"
                                                for lid in langids]) if langids else None,
                              postags=" ".join(postags) if postags else None,
                              nertags=" ".join(nertags) if nertags else None)
            examples.append(example)

            # because test does not have a next line as `# sent_enum = xx`
            if mode == "test":
                label = None
                uid = None
                tokens, langids, postags, nertags = [], [], [], []
        elif "# sent_enum =" in line:
            # start a new line and reset field values
            vals = line.strip().split("\t")
            label = vals[-1] if len(vals) > 1 else None
            uid = vals[0].split("=")[-1].strip()
            tokens, langids, postags, nertags = [], [], [], []
        else:
            vals = line.strip().split("\t")
            if not mode == "test":
                tokens.append(vals[0])
                langids.append(vals[1])
                if is_pos:
                    postags.append(vals[2])
                elif is_ner:
                    nertags.append(vals[2])
            else:
                tokens.append(vals[0])
                if is_pos or is_ner:
                    langids.append(vals[1])
        progress_bar(line_num, len(all_lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_gluecos_downloads(path, mode, dataset_name, task_name, is_pos=False, is_ner=False,
                           standardizing_tags={}):
    st_time = time()

    global FIELDS
    tokens, langids, postags, nertags, label, uid = [], [], [], [], None, None
    FIELDS += [fieldname for fieldname in ("nertags", "postags") if fieldname not in FIELDS]
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    all_lines = open(path, "r").readlines()
    for line_num, line in enumerate(all_lines):
        if line.strip() == "":
            if not tokens:
                continue
            uid = len(examples)
            tokens = ["".join(tkn.split()) for tkn in tokens]
            txt = " ".join(tokens)
            new_txt = clean_generic(txt)
            if new_txt.strip() == "":
                new_txt = txt
            example = Example(dataset=dataset_name,
                              task=task_name,
                              split_type=mode,
                              uid=uid,
                              label=label,
                              text=txt,
                              text_pp=new_txt,
                              langids=" ".join([standardizing_tags[lid] if lid in standardizing_tags else "other"
                                                for lid in langids]) if langids else None,
                              postags=" ".join(postags) if postags else None,
                              nertags=" ".join(nertags) if nertags else None)
            examples.append(example)
            tokens, langids, postags, nertags, label, uid = [], [], [], [], None, None
        else:
            vals = line.strip().split("\t")
            if not mode == "test":
                if is_pos:
                    if len(vals) < 3:
                        continue
                    tokens.append(vals[0])
                    langids.append(vals[1])
                    postags.append(vals[2])
                elif is_ner:
                    if len(vals) < 2:
                        continue
                    tokens.append(vals[0])
                    nertags.append(vals[1])
            else:
                tokens.append(vals[0])
                if is_pos:
                    langids.append(vals[1])
        progress_bar(line_num, len(all_lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)}")
    return examples


def read_vsingh_downloads(path, mode="train"):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN
    add_fields = {
    }
    for k, v in add_fields.items():
        FIELDS += (vv for vv in v if vv not in FIELDS)

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    all_lines = open(path, "r").readlines()
    tokens_dict, tags_dict = {}, {}
    for line_num, line in enumerate(all_lines):
        line = line.strip()
        if line_num == 0 or "sent" not in line:
            continue
        line_tokens = line.split(",")
        if '","' in line:
            _id_info, word, tag = line_tokens[0], ",", line_tokens[-1]
        else:
            _id_info, word, tag = line_tokens[0], line_tokens[1], line_tokens[-1]
        _id = _id_info[6:]
        if not _id in tokens_dict:
            tokens_dict[_id], tags_dict[_id] = [], []
        tokens_dict[_id].append(word)
        tags_dict[_id].append(tag)

    for _id in tokens_dict:
        txt = " ".join(tokens_dict[_id])
        tags = " ".join(tags_dict[_id])
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            # new_txt = txt
            continue
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="vsinghetal_2018",
                          task="seq_tagging",
                          split_type=mode,
                          uid=_id,
                          text=txt,
                          langids=tags,
                          text_pp=new_txt)
        examples.append(example)
        progress_bar(len(examples), len(tokens_dict), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_mt1_downloads(path1, path2, mode, dataset_name, task_name):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    n_trimmed = 0
    FIELDS += [fieldname for fieldname in ["tgt", "tgt_pp", ] if fieldname not in FIELDS]
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    txt_lines = [line.strip() for line in open(path1, "r")]
    tgt_lines = [line.strip() for line in open(path2, "r")]
    for i, (txt, tgt) in enumerate(zip(txt_lines, tgt_lines)):
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        new_tgt = clean_generic(tgt)
        if new_tgt.strip() == "":
            new_tgt = tgt
        if len(new_tgt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_tgt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_tgt = " ".join(newtokens)
        example = Example(dataset=dataset_name,
                          task=task_name,
                          split_type=mode,
                          uid=i,
                          text=txt,
                          text_pp=new_txt,
                          tgt=tgt,
                          tgt_pp=new_tgt)
        examples.append(example)
        progress_bar(len(examples), len(txt_lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_mt2_downloads(path, mode, dataset_name, task_name):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    n_trimmed = 0
    FIELDS += [fieldname for fieldname in ["tgt", "tgt_pp", ] if fieldname not in FIELDS]
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    rows = read_csv_file(path)
    txt_lines = [line[0].strip() for line in rows]
    tgt_lines = [line[1].strip() for line in rows]
    for i, (txt, tgt) in enumerate(zip(txt_lines, tgt_lines)):
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        new_tgt = clean_generic(tgt)
        if new_tgt.strip() == "":
            new_tgt = tgt
        if len(new_tgt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_tgt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_tgt = " ".join(newtokens)
        example = Example(dataset=dataset_name,
                          task=task_name,
                          split_type=mode,
                          uid=i,
                          text=txt,
                          text_pp=new_txt,
                          tgt=tgt,
                          tgt_pp=new_tgt)
        examples.append(example)
        progress_bar(len(examples), len(txt_lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_royetal2013_downloads(path, mode, standardizing_tags={}):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    lines = [line.strip() for line in open(path, "r")]
    for i, line in enumerate(lines):
        tkns = line.split()
        txt = " ".join([tkn.split("\\")[0].strip() for tkn in tkns])
        tgs = " ".join([tkn.split("\\")[1].split("=")[0].strip() for tkn in tkns])
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="royetal2013_lid",
                          task="classification",
                          split_type=mode,
                          uid=i,
                          text=txt,
                          langids=" ".join([standardizing_tags[lid] if lid in standardizing_tags else "other"
                                            for lid in tgs.split(" ")]),
                          text_pp=new_txt)
        examples.append(example)
        progress_bar(len(examples), len(lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_semeval2017_en_sa_downloads(path, mode):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    lines = [line.strip() for line in open(path, "r")]
    for i, line in enumerate(lines):
        try:
            uid, label, txt = line.split("\t")[:3]
        except Exception as e:
            print(path)
            print(line)
            print(line.split("\t"))
            # raise Exception
            continue
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="semeval2017_en_sa",
                          task="classification",
                          split_type=mode,
                          uid=uid,
                          text=txt,
                          label=label,
                          text_pp=new_txt)
        examples.append(example)
        progress_bar(len(examples), len(lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_iitp_product_reviews_hi_sa_downloads(path, mode):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    from indictrans import Transliterator
    trn_hin2eng = Transliterator(source='hin', target='eng')

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    lines = [line.strip() for line in open(path, "r")]
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        vals = line.split(",")
        label = vals[0]
        txt = ",".join(vals[1:])
        txt = trn_hin2eng.transform(txt)
        new_txt = "".join([char for char in txt])
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="iitp_product_reviews_hi_sa",
                          task="classification",
                          split_type=mode,
                          uid=len(examples),
                          text=txt,
                          label=label,
                          text_pp=new_txt)
        examples.append(example)
        progress_bar(len(examples), len(lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_sentimix2020_downloads(path, mode="train", test_labels: dict = {}, standardizing_tags={}):
    st_time = time()
    to_translate = []

    global FIELDS, MAX_CHAR_LEN
    add_fields = {
        # "noeng": ['text_noeng'],
        # "nohin": ["text_nohin"],
        # "trt": ['text_trt'],
        # "trt_noeng": ["text_trt_noeng"],
        # "fd": ["text_fd"],
    }
    for k, v in add_fields.items():
        FIELDS += (vv for vv in v if vv not in FIELDS)

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    uid, label, tokens, lang_ids = None, None, None, None
    start_new = True
    lines = open(path, 'r').readlines()
    for idx, line in enumerate(lines):
        if line == "\n" or len(lines) - 1 == idx and tokens:
            org_tokens = [token for token in tokens]
            org_tags = [langid for langid in lang_ids]
            assert len(org_tokens) == len(org_tags), print(len(org_tokens), len(org_tags))
            tokens, lang_ids = clean_sentimix2020_lines(org_tokens, org_tags)
            if " ".join(tokens).strip() == "":
                tokens, lang_ids = org_tokens, org_tags
            # if len(" ".join(tokens)) > MAX_CHAR_LEN:
            #     n_trimmed += 1
            #     newtokens, newlangids, currsum = [], [], 0
            #     for tkn, lgid in zip(tokens, lang_ids):  # 1 for space
            #         if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
            #             newtokens.append(tkn)
            #             newlangids.append(lgid)
            #             currsum += len(tkn) + 1
            #         else:
            #             break
            #     tokens, lang_ids = newtokens, newlangids
            example = Example(dataset="sentimix2020_hinglish",
                              task="classification",
                              split_type=mode,
                              uid=uid,
                              text=" ".join(org_tokens),
                              label=label,
                              langids=" ".join(
                                  [standardizing_tags[lid] if lid in standardizing_tags else "other" for lid in
                                   org_tags]),
                              text_pp=" ".join(tokens),
                              langids_pp=" ".join(lang_ids))

            examples.append(example)
            progress_bar(len(examples), 14000 if mode == "train" else 3000, ["time"], [time() - st_time])
            start_new = True
            continue
        parts = line.strip().split("\t")
        if start_new:
            uid, label = (parts[1], parts[2]) if mode != "test" else (parts[1], test_labels.get(parts[1], None))
            tokens, lang_ids = [], []
            start_new = False
        else:
            if len(parts) == 2:  # some lines are weird, e.g see uid `20134`
                tokens.append(parts[0])
                lang_ids.append(parts[1])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")

    def my_repl(matchobj):
        ans = response_translations[int(matchobj.group(0).split("_")[-1]) - 1]
        return ans

    if "fd" in add_fields:
        response_translations = []
        for i in range(int(len(to_translate) / 1024) + 1):
            response_translations.extend(google_translate(to_translate[i * 1024: (i + 1) * 1024], "hi"))
        assert (len(to_translate) == len(response_translations))
        new_examples = []
        for example in examples:
            new_text_fd = re.sub("UNIQUE_TRANSLATION_ID_\d{1,}", my_repl, example.text_fd)
            example = example._replace(text_fd=new_text_fd)
            new_examples.append(example)
        examples = new_examples

    return examples


def read_sail2017_downloads(path, mode="train"):
    st_time = time()
    to_translate = []

    global FIELDS, MAX_CHAR_LEN
    add_fields = {
    }
    for k, v in add_fields.items():
        FIELDS += (vv for vv in v if vv not in FIELDS)

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    all_lines = open(path, "r").readlines()
    for uid, line in enumerate(all_lines):
        if line.strip() == "":
            continue
        txt, label = [part.strip() for part in line.strip().split("\t")][:2]
        new_txt = clean_sail2017_lines(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="sail_2017_hinglish",
                          task="classification",
                          split_type=mode,
                          uid=uid,
                          text=txt,
                          text_pp=new_txt,
                          label=label)
        examples.append(example)
        progress_bar(len(examples), len(all_lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_sail2017_downloads_new(path, labelled_dict, mode="train", standardizing_tags={}):
    st_time = time()
    to_translate = []
    label_dict = {"-1": "negative", "0": "neutral", "1": "positive"}

    global FIELDS, MAX_CHAR_LEN
    add_fields = {
    }
    for k, v in add_fields.items():
        FIELDS += (vv for vv in v if vv not in FIELDS)
    not_found = 0

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    all_lines = open(path, "r").readlines()
    for line in all_lines:
        _id = line.strip()
        if _id not in labelled_dict:
            not_found += 1
            continue
        else:
            data = labelled_dict[_id]
            txt, txt_langids, label = data["text"], data["lang_tagged_text"], label_dict[str(data["sentiment"])]
        org_tokens, org_tags = [], []
        for x in txt_langids.split(" "):
            if x:
                vals = x.split("\\")
                org_tokens.append(vals[0])
                org_tags.append(vals[-1])
        assert len(org_tokens) == len(org_tags), print(len(org_tokens), len(org_tags))
        tokens, lang_ids = clean_sail2017_lines(org_tokens, org_tags)
        if " ".join(tokens).strip() == "":
            tokens, lang_ids = org_tokens, org_tags
        # if len(" ".join(tokens)) > MAX_CHAR_LEN:
        #     n_trimmed += 1
        #     newtokens, newlangids, currsum = [], [], 0
        #     for tkn, lgid in zip(tokens, lang_ids):  # 1 for space
        #         if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
        #             newtokens.append(tkn)
        #             newlangids.append(lgid)
        #             currsum += len(tkn) + 1
        #         else:
        #             break
        #     tokens, lang_ids = newtokens, newlangids
        example = Example(dataset="sail2017_hinglish",
                          task="classification",
                          split_type=mode,
                          uid=_id,
                          text=" ".join(org_tokens),
                          langids=" ".join([standardizing_tags[lid] if lid in standardizing_tags else "other"
                                            for lid in org_tags]),
                          text_pp=" ".join(tokens),
                          langids_pp=" ".join(lang_ids),
                          label=label)
        examples.append(example)
        progress_bar(len(examples), len(all_lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed} "
          f"and # of not found instances: {not_found}")
    return examples


def read_subwordlstm_downloads(path, mode="train"):
    st_time = time()
    to_translate = []
    label_dict = {"0": "negative", "1": "neutral", "2": "positive"}

    global FIELDS, MAX_CHAR_LEN
    add_fields = {
    }
    for k, v in add_fields.items():
        FIELDS += (vv for vv in v if vv not in FIELDS)

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    all_lines = open(path, "r").readlines()
    for uid, line in enumerate(all_lines):
        if line.strip() == "":
            continue
        txt, label = [part.strip() for part in line.strip().split("\t")][:2]
        label = label_dict[label]
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="subwordlstm_2016_hinglish",
                          task="classification",
                          split_type=mode,
                          uid=uid,
                          text=txt,
                          text_pp=new_txt,
                          label=label)
        examples.append(example)
        progress_bar(len(examples), len(all_lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_hinglishpedia_downloads(path1, path2, mode, standardizing_tags={}):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    n_trimmed = 0
    FIELDS += [fieldname for fieldname in ["tgt", ] if fieldname not in FIELDS]
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []

    from indictrans import Transliterator
    trn_hin2eng = Transliterator(source='hin', target='eng')

    txt_lines = [line.strip() for line in open(path1, "r")]
    tag_lines = [line.strip() for line in open(path2, "r")]

    for i, (txt, tags) in tqdm(enumerate(zip(txt_lines, tag_lines))):
        if not txt:
            continue
        txt = trn_hin2eng.transform(txt)
        example = Example(dataset="hinglishpedia",
                          task="classification",
                          split_type=mode,
                          uid=len(examples),
                          text=txt,
                          langids=" ".join([standardizing_tags[lid] if lid in standardizing_tags else "other"
                                            for lid in tags.split()]),
                          text_pp=txt
                          )
        examples.append(example)
        # progress_bar(len(examples), len(txt_lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_kumaretal_2019_agg_downloads(path, mode, romanize=False):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    from indictrans import Transliterator
    trn_hin2eng = Transliterator(source='hin', target='eng')

    n_trimmed, n_romanized = 0, 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    lines = read_csv_file(path, has_header=False)
    for i, line in enumerate(lines):
        uid, txt, label = line[0], line[1], line[2]
        if not txt:
            continue
        if romanize:
            new_txt = trn_hin2eng.transform(txt)
            if txt != new_txt:
                n_romanized += 1
            txt = new_txt
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="kumaretal_2019_agg",
                          task="classification",
                          split_type=mode,
                          uid=uid,
                          text=txt,
                          label=label,
                          text_pp=new_txt)
        examples.append(example)
        progress_bar(len(examples), len(lines), ["time"], [time() - st_time])

    if romanize:
        print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed} "
              f"and # of romanized instances: {n_romanized}")
    else:
        print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")

    return examples


def read_kumaretal_2020_agg_downloads(path, mode, test_labels_file=None, romanize=False):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    from indictrans import Transliterator
    trn_hin2eng = Transliterator(source='hin', target='eng')

    if mode == "test":
        test_label_lines = read_csv_file(test_labels_file, has_header=True)

    n_trimmed, n_romanized = 0, 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    lines = read_csv_file(path, has_header=True)
    for i, line in enumerate(lines):
        if mode == "test":
            uid, txt = line[0], line[1]
            assert test_label_lines[i][0] == uid
            label = test_label_lines[i][1]
        else:
            uid, txt, label = line[0], line[1], line[2]
        if not txt:
            continue
        if romanize:
            new_txt = trn_hin2eng.transform(txt)
            if txt != new_txt:
                n_romanized += 1
            txt = new_txt
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="kumaretal_2020_agg",
                          task="classification",
                          split_type=mode,
                          uid=uid,
                          text=txt,
                          label=label,
                          text_pp=new_txt)
        examples.append(example)
        progress_bar(len(examples), len(lines), ["time"], [time() - st_time])

    if romanize:
        print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed} "
              f"and # of romanized instances: {n_romanized}")
    else:
        print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")

    return examples


def read_vijayetal_2018_hatespeech_downloads(path, mode):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    lines = read_csv_file(path, has_header=False, delimiter="\t")
    for i, line in enumerate(lines):
        uid, txt, label = len(examples), line[0].strip(), line[1].strip()
        if label == "n" or label == "on":
            label = "no"
        assert label in ["yes", "no"], print(label)
        if not txt:
            continue
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="vijayetal_2018_hatespeech",
                          task="classification",
                          split_type=mode,
                          uid=uid,
                          text=txt,
                          label=label,
                          text_pp=new_txt)
        examples.append(example)
        progress_bar(len(examples), len(lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")

    return examples


def read_kauretal_2019_reviews_downloads(path, mode):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    label_def = {
        1: "Gratitude",
        2: "About the recipe",
        3: "About the video",
        4: "Praising",
        5: "Hybrid",
        6: "Undefined",
        7: "Suggestions and queries"
    }

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    lines = read_csv_file(path, has_header=True)
    for i, line in enumerate(lines):
        uid, txt, label = line[0], line[1], label_def[int(line[2].strip())]
        if not txt:
            continue
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="kauretal_2019_reviews",
                          task="classification",
                          split_type=mode,
                          uid=uid,
                          text=txt,
                          label=label,
                          text_pp=new_txt)
        examples.append(example)
        progress_bar(len(examples), len(lines), ["time"], [time() - st_time])

    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")

    return examples


def create_data_for_adaptation(data_folders: List[str], dest_folder: str, base_data_folder="", key_field="text"):
    print("prepare data for adaptation...")
    max_text_length = 300
    if base_data_folder:
        data_folders = [os.path.join(base_data_folder, folder) for folder in data_folders]
    create_path(dest_folder)

    # train data
    train_lines = []
    n_found_train, n_found_dev = 0, 0
    for check_dir in data_folders:
        filenames = os.listdir(check_dir)
        if "train.jsonl" in filenames:
            n_found_train += 1
            train_examples = read_datasets_jsonl(os.path.join(check_dir, "train.jsonl"))
            train_lines.extend([getattr(ex, key_field) for ex in train_examples])
        else:
            raise Exception("Could not find `train.jsonl` file in the dataset")
        if "test.jsonl" in filenames and "dev.jsonl" in filenames:
            n_found_dev += 1
            dev_examples = read_datasets_jsonl(os.path.join(check_dir, "dev.jsonl"))
            train_lines.extend([getattr(ex, key_field) for ex in dev_examples])
    print(f"train files found in {n_found_train} datasets from the given {len(data_folders)} datasets")
    print(f"dev files found in {n_found_dev} datasets from the given {len(data_folders)} datasets")
    print(f"total train lines obtained (before any further processing): {len(train_lines)}")
    opfile = open(os.path.join(dest_folder, "train.txt"), "w")
    c, cc = 0, 0
    for line in train_lines:
        line = line.strip()
        if not line:
            continue
        if len(line) < max_text_length:
            c += 1
            opfile.write(f"{line}\n")
        else:
            curr_len, curr_tokens = 0, []
            tokens = [tkn + " " for tkn in line.split()]
            tokens[-1] = tokens[-1][:-1]
            for token in tokens:
                if curr_len + len(token) > max_text_length:
                    sub_line = "".join(curr_tokens)
                    assert len(sub_line) <= max_text_length, print(len(sub_line), sub_line)
                    opfile.write(f"{sub_line}\n")
                    cc += 1
                    if len(token) > max_text_length:
                        break
                    curr_len, curr_tokens = 0, []
                curr_tokens.append(token)
                curr_len += len(token)
    opfile.close()
    print(f"total train lines written: (as-is plus chunked): {c}+{cc}={c + cc}")

    # test data
    test_lines = []
    n_found = 0
    for check_dir in data_folders:
        filenames = os.listdir(check_dir)
        if "test.jsonl" in filenames:
            n_found += 1
            test_examples = read_datasets_jsonl(os.path.join(check_dir, "test.jsonl"))
            test_lines.extend([getattr(ex, key_field) for ex in test_examples])
        elif "dev.jsonl" in filenames:
            n_found += 1
            test_examples = read_datasets_jsonl(os.path.join(check_dir, "dev.jsonl"))
            test_lines.extend([getattr(ex, key_field) for ex in test_examples])
        else:
            raise Exception("No suitable test file found. Tried looking for `dev.jsonl` and `test.jsonl`")
    print("")
    print(f"test files found in {n_found} datasets from the given {len(data_folders)} datasets")
    print(f"total test lines obtained: {len(test_lines)}")
    opfile = open(os.path.join(dest_folder, "test.txt"), "w")
    c, cc = 0, 0
    for line in test_lines:
        line = line.strip()
        if not line:
            continue
        if len(line) < max_text_length:
            opfile.write(f"{line}\n")
            c += 1
        # else:
        #     curr_len, curr_tokens = 0, []
        #     tokens = [tkn+" " for tkn in line.split()]
        #     tokens[-1] = tokens[-1][:-1]
        #     for token in tokens:
        #         if curr_len + len(token) > max_text_length:
        #             sub_line = "".join(curr_tokens)
        #             assert len(sub_line) <= max_text_length, print(len(sub_line), sub_line)
        #             opfile.write(f"{sub_line}\n")
        #             cc += 1
        #             if len(token) > max_text_length:
        #                 break
        #             curr_len, curr_tokens = 0, []
        #         curr_tokens.append(token)
        #         curr_len += len(token)
    opfile.close()
    print(f"total test lines written: (as-is plus chunked):  {c}+{cc}={c + cc}")

    print(f"adaptation data saved at: {dest_folder}")
    return
