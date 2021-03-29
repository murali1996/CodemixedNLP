# import spacy
# import string
# import itertools
import copy
import re
import unicodedata
from typing import List

from wordsegment import load as wordsegmentload
from wordsegment import segment as wordsegmenter

wordsegmentload()


def space_around_punct(text: str):
    return re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", text).strip()


def remove_character_repetitions(text: str):
    return re.sub(r"(.)\1{1,}", r"\1\1", text)


def clean_sail2017_lines(tokens: List[str], lang_ids: List[str]):
    tokens, lang_ids = [token for token in tokens], [_id for _id in lang_ids]
    i = 0
    while i < len(tokens):
        text, tag = tokens[i], lang_ids[i]
        if text == "+" and " ".join(text[i:i + 4]) == "+ en _ suffix":
            i = i + 4
            continue
        i += 1
    tokens, lang_ids = clean_sentimix2020_lines(tokens, lang_ids)
    assert len(tokens) == len(lang_ids), print(len(tokens), len(lang_ids))
    return tokens, lang_ids


def clean_generic(text: str):
    new_text = copy.deepcopy(text)
    if new_text[:3] == "RT ":
        new_text = new_text[3:]
    new_text = _normalize(new_text)
    new_text = _remove_html_tags(new_text)
    new_text = _remove_generic_https(new_text)
    new_text = re.sub("pic\.twitter\.com/\w+", "", new_text)
    new_text = _remove_generic_misc(new_text)
    new_text = remove_character_repetitions(new_text)
    new_text = space_around_punct(new_text)
    return new_text


def clean_sentimix2020_lines(tokens: List[str], lang_ids: List[str]):
    tokens, lang_ids = [token for token in tokens], [_id for _id in lang_ids]
    if tokens[0].lower() == "rt":
        tokens = tokens[1:]
        lang_ids = lang_ids[1:]
    tokens, lang_ids = _remove_sentimix_https(tokens, lang_ids)
    tokens, lang_ids = _remove_sentimix_nametags(tokens, lang_ids)
    # tokens_plus_ids = [_camel_case_split(token, lang_id)
    #                    if (lang_id.lower() == "eng" or lang_id.lower() == "en") else (token, lang_id)
    #                    for (token, lang_id) in zip(tokens, lang_ids)]
    # tokens, lang_ids = list(zip(*tokens_plus_ids))
    # tokens, lang_ids = " ".join(tokens).split(" "), " ".join(lang_ids).split(" ")
    tokens, lang_ids = _remove_character_repetitions_wtags(tokens, lang_ids)
    tokens, lang_ids = _space_around_punct_wtags(tokens, lang_ids)
    tokens, lang_ids = _normalize_wtags(tokens, lang_ids)
    assert len(tokens) == len(lang_ids), print(len(tokens), len(lang_ids))
    return tokens, lang_ids


def _remove_sentimix_nametags(tokens: List[str], tags: List[str]):
    new_tokens, new_tags = [], []
    check = True
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if not check:
            check = False if ((i + 1 < len(tokens) and "_" in tokens[i + 1]) or "_" in token) else True
            continue
        if "@" in token and i + 1 < len(tokens):
            check = False
            continue
        else:
            new_tokens.append(token)
            new_tags.append(tag)
    if len(new_tokens) == 0:
        new_tokens, new_tags = tokens, tags
    return new_tokens, new_tags


def _remove_character_repetitions_wtags(tokens: List[str], tags: List[str]):
    new_tokens, new_tags = [], []
    for token, tag in zip(tokens, tags):
        new_token = re.sub(r"(.)\1{1,}", r"\1\1", token)
        if new_token:
            new_tokens.append(new_token)
            new_tags.append(tag)
    if len(new_tokens) == 0:
        new_tokens, new_tags = tokens, tags
    return new_tokens, new_tags


def _camel_case_split(eng_word, tag, verbose=True):
    if verbose and (tag.lower() != "eng" and tag.lower() != "en" and tag.lower() != ""):
        print(f"_camel_case_split() called for non english tag. tag::{tag} || word::{eng_word}")
    new_eng_word = re.sub("(\w+)(vs)(\w+)", r"\1 \2 \3", eng_word)
    if len(new_eng_word.split()) > 1:
        subwords = new_eng_word.split()
    else:
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', eng_word)
        subwords = [m.group(0) for m in matches]
    tag = [tag] * len(subwords)
    return " ".join(subwords), " ".join(tag)


def _word_segment_me(eng_word: str, tag: str, verbose=True):
    if verbose and (tag.lower() != "eng" and tag.lower() != "en" and tag.lower() != ""):
        print(f"_word_segment_me() called for non english tag. tag::{tag} || word::{eng_word}")
    subwords = wordsegmenter(eng_word)
    if eng_word.startswith("#"):
        subwords = ["#"] + subwords  # probably retaining that it is a hash context can be useful
    tag = [tag] * len(subwords)
    return " ".join(subwords), " ".join(tag)


def _normalize(text: str):
    text = text.encode("ascii", "ignore").decode().strip()
    text = unicodedata.normalize('NFD', text)
    return text


def _normalize_wtags(tokens: List[str], tags: List[str]):
    new_tokens, new_tags = [], []
    for token, tag in zip(tokens, tags):
        new_token = _normalize(token)
        if new_token:
            new_tokens.append(new_token)
            new_tags.append(tag)
    if len(new_tokens) == 0:
        new_tokens, new_tags = tokens, tags
    return new_tokens, new_tags


def _space_around_punct_wtags(tokens: List[str], tags: List[str]):
    new_tokens, new_tags = [], []
    for token, tag in zip(tokens, tags):
        subs = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)(\s*)", r"\1 ", token).strip().split()
        new_tokens.extend(subs)
        new_tags.extend([tag] * len(subs))
    assert len(new_tokens) == len(new_tags), print(len(new_tokens), len(new_tags))
    return new_tokens, new_tags


def _remove_html_tags(new_line: str):
    new_line = re.sub("(?:&\w+;\d|&\w+;)", "", new_line)
    new_line = re.sub('\s{2,}', ' ', new_line)
    return new_line.strip()


def _remove_generic_https(new_line: str):
    # new_line = re.sub("(?:&amp;|&amp)", "", new_line)
    new_line = re.sub("(?:https|http)\S+", "", new_line)
    new_line = " ".join([word for word in new_line.split(" ") if word != "http"])
    new_line = re.sub('\s{2,}', ' ', new_line)
    return new_line


def _remove_generic_misc(new_line: str):
    new_line = re.sub("@\S+", "", new_line)
    new_line = re.sub("(\+en_suffix)", "", new_line)
    new_line = re.sub("#(\w+)(vs)(\w+)", r"# \1 \2 \3", new_line)
    new_line = re.sub("#(\w+)(v)(\w+)", r"# \1 \2 \3", new_line)
    new_line = re.sub("(\. ){1,}\.", "...", new_line)
    new_line = " ".join([" ".join(word.split("_"))[1:] if (word.startswith("#") and "_" in word) else word
                         for word in new_line.split(" ")])
    # for words starting with #, first try camel case split, if that doesn't work, then only go for word segmentation
    new_tokens = []
    for token in new_line.strip().split():
        newtoken = token
        if token.startswith("#") and len(token) > 1:
            newtoken, _ = _camel_case_split(token, "")
            if newtoken == token:
                newtoken, _ = _word_segment_me(token, "", verbose=False)
        new_tokens.append(newtoken)
    new_line = " ".join(new_tokens)
    # new_line = " ".join([_word_segment_me(token, "", verbose=False)[0] if token.startswith("#") else token
    #                      for token in new_line.strip().split()])
    return new_line


def _remove_sentimix_https(tokens, tags):
    """
    :param tokens: a sequence of text that can be split at white-space
    :param tags: a sequence of tags, one tag corresponding to one word in text
    :return: processed_text_line along with corresponding (modified) tags line
    """
    # text_line = re.sub('https // tco / [a-zA-Z0-9]{10,}', '', text_line)
    # text_line = re.sub('https // t co / [a-zA-Z0-9]{10,}', '', text_line)
    # text_line = re.sub('https // t . co / [a-zA-Z0-9]{10,}', '', text_line)
    # text_line = re.sub('https // tco / [a-zA-Z0-9]{3,}', '', text_line)
    # new
    # re.sub("http :// t . co / izwquzsi2h", '', text_line)
    # re.sub("http / URL", '', text_line)
    # return text_line

    text_tokens, tag_tokens = [token for token in tokens], [tag for tag in tags]
    if len(text_tokens) != len(tag_tokens):
        print(len(text_tokens), len(tag_tokens))
        raise Exception
    new_text_tokens, new_tag_tokens = [], []
    next_i = -1
    for i, (token, tag) in enumerate(zip(text_tokens, tag_tokens)):
        if i < next_i:
            continue
        if (token == "https" or token == "http") and \
                (text_tokens[i + 1] == "//" or text_tokens[i + 1] == "/" or text_tokens[i + 1] == "://"):
            if text_tokens[i + 2] == "tco":
                next_i = i + 5
            elif text_tokens[i + 2] == "t" and text_tokens[i + 3] == "co":
                next_i = i + 6
            elif text_tokens[i + 2] == "t" and text_tokens[i + 3] == ".":
                next_i = i + 7
            elif text_tokens[i + 2].lower() == "url":
                next_i = i + 3
            else:
                new_text_tokens.append(token)
                new_tag_tokens.append(tag)
        else:
            new_text_tokens.append(token)
            new_tag_tokens.append(tag)
    return new_text_tokens, new_tag_tokens


# if __name__ == "__main__":
#     print()
#
#     def data_preprocess(lines: list):
#         lines = [line.lower() for line in lines]
#
#         # lines = [re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+\S+', "", line) for line in lines]
#         # lines = [re.sub('http?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+\S+', "", line) for line in lines]
#         lines = [re.sub(r'https?:\/\/.*[\r\n]*', '', line) for line in lines]
#         lines = [re.sub(r"<([^>]+)>", "", line) for line in lines]
#
#         # lines = [unidecode.unidecode(line) for line in lines]
#         lines = [line.encode('unicode_escape').decode().replace('\\\\', ' \\') for line in lines]
#         lines = [re.sub(r"\\\S+", "", line) for line in lines]
#         lines = [re.sub(r"\\", "", line) for line in lines]
#
#         lines = [" ".join(line.split(";")) for line in lines]
#         lines = [" ".join(line.split(".")) for line in lines]
#
#         # lines = [" ".join(line.split("@")) for line in lines]
#         lines = [re.sub(r"(@\S+)", "", line) for line in lines]
#         lines = [re.sub(r"(&\S+)", "", line) for line in lines]  # tokens like &amp &gt
#
#         # "".join(re.compile('(\w+)').findall(word))
#         segmentme = lambda word: " ".join(wordsegmenter(word)) if word.startswith("#") else word
#         lines = [" ".join([segmentme(word) for word in line.split()]) for line in lines]
#
#         # temp_lines = []
#         # for line in lines:
#         #     for key, val in contraction_mapping.items():
#         #         line = line.replace(key, val + " ")
#         #     temp_lines.append(line)
#         # lines = temp_lines
#
#         spacy_tokenizer_ = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
#         spacy_tokenizer = lambda lines: [" ".join([token.text for token in spacy_tokenizer_(inp)]) for inp in lines]
#         lines = spacy_tokenizer(lines)
#         ispunct = lambda word: len([char for char in word if char in string.punctuation]) == len(word)
#         lines = [" ".join([word for word in line.split() if not ispunct(word)]) for line in lines]
#
#         # lines = [" ".join([word for word in line.split() if not word.isdigit()]) for line in lines]
#         lines = [re.sub(r"\b\d\d+\b", "", line) for line in lines]
#
#         lines = [''.join(''.join(s)[:2] for _, s in itertools.groupby(line)) for line in lines]
#
#         return lines
#
#     texts = ["@ hmmmmmmmmm got didn't didnt some antifa  to share?  i'm interested \\ud83dhttp://t.co/A2RbVEBeg1",
#              "#doesn'tunderstandmetaphors http://t.co/A2RbVEBeg1\\ud83 http://t.co/A2RbVEBeg1 \\ud83",
#              "#doesntunderstandmetaphorshaha",
#              "!!!!&#8220;@selfiequeenbri: cause I'm tired of you big bitches coming for us skinny girls!!&#8221;",
#              "#VoteRepublican♨️#ARM♨️#AmericanRedMidterms http://t.co/A2RbVEBeg1 http://t.co/A2RbVEBeg1"" "
#              "@384708737 $263 873 93743mm379",
#              '#trumpvoters#helloworlds f***ck fuckiiiiing shhhh hmmmmm @user @user @user '
#              '@user fascists &amp;nazis don\'t abide by patriotic views shared in their presence. '
#              'glad the brownshirt arm (antifa) of the left wasn\'t there to inflict the usual violence '
#              'vs citizens who dare to speak. do you hear that, dems who enjoy wings/beer/country music? '
#              'you\'re less-than." #walkaway"',
#              "#trumpnation",
#              "@USER stop... /xhghkv f**k n**ga* Don't talk or think about it... Never shoula "
#              "watched it! \ud83d\ude37	Like I'm a girl Idfk what to do to get my car fixed hahaha",
#              "fuck he is... kl has a bigger dick than me but she’s a she...he🤔...she?!? url",
#              "\\ud83d\\ude02\\ud83d\\ude02\\ud83d\\ude02\\ud83d\\ude02",
#              "https hello"]
#
#     for text in data_preprocess(texts):
#         print(text)
#
#     text1 = 'karoge … https // t . co / 1pQz7BQlcG'
#     text2 = 'hello https // t . co / 1pQz7BQlcG testing https // none https t'
#     text3 = 'hello https // t co / 1pQz7BQlcG testing https // none https t'
#     text4 = 'hello https // tco / 1pQz7BQlcG testing https // none https t'
#     text5 = 'hello https // tco / 1pQz7BQlcG'
#     text6 = 'hello https // tco / aaaaa testing https // none https t'
#     for text in [text1, text2, text3, text4, text5, text6]:
#         tags = " ".join([str(x) for x in range(len(text.split()))])
#         print(f"{text}\n{tags}\n{remove_sentimix_https(text, tags)}\n")
