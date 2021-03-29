import os

SRC_ROOT_PATH = os.path.split(__file__)[0]  # inside-CodemixedNLP directory

CHECKPOINTS_PATH = os.path.join(os.path.split(__file__)[0], "../checkpoints")
# print(f"\ncheckpoints folder set to: \n`{CHECKPOINTS_PATH}` in `CodemixedNLP/paths.py` script\n")

ARXIV_CHECKPOINTS = {
    "sentiment": os.path.join(CHECKPOINTS_PATH, "arxiv-sentiment/xlm-roberta-base"),
    "aggression": os.path.join(CHECKPOINTS_PATH, "arxiv-aggression/xlm-roberta-base"),
    "hatespeech": os.path.join(CHECKPOINTS_PATH, "arxiv-hatespeech/xlm-roberta-base"),
    "lid": os.path.join(CHECKPOINTS_PATH, "arxiv-lid/xlm-roberta-base"),
    "pos": os.path.join(CHECKPOINTS_PATH, "arxiv-pos/xlm-roberta-base"),
    "ner": os.path.join(CHECKPOINTS_PATH, "arxiv-ner/xlm-roberta-base"),
    "mt": os.path.join(CHECKPOINTS_PATH, "arxiv-mt/hing-eng/VanillaTransformerModel"),
}
