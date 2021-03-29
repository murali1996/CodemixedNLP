#############################################
# USAGE
# -----
# python download_checkpoints.py
#############################################

# taken from https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python
# taken from this StackOverflow answer: https://stackoverflow.com/a/39225039

import os

import requests

from ..paths import ARXIV_CHECKPOINTS


def _download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, destination)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def _create_paths(path_: str):
    if not os.path.exists(path_):
        os.makedirs(path_)
        print(f"{path_} created")
    else:
        print(f"{path_} already exists; skipping downloading model")
        return False
    return True


def download_sentiment():
    print("downloading model for `sentiment` classification")
    save_path = ARXIV_CHECKPOINTS["sentiment"]
    if _create_paths(save_path):
        _download_file_from_google_drive('1mSqDUV1GnhSaVzqnyX1x1GazeFQJk7n6', os.path.join(save_path, "model.pth.tar"))
        _download_file_from_google_drive('18uF40MIOD_KixXusJP848xp8JLT1ijV-',
                                         os.path.join(save_path, "label_vocab.json"))
    return


def download_aggression():
    print("downloading model for `aggression` classification")
    save_path = ARXIV_CHECKPOINTS["aggression"]
    if _create_paths(save_path):
        _download_file_from_google_drive('1exeRlVcj3b5UrQMCIHzqq0Wl-ZDNXY6m', os.path.join(save_path, "model.pth.tar"))
        _download_file_from_google_drive('1Tl8p236P9GAzqVFAkWY3Xt9RG8wVlhMv',
                                         os.path.join(save_path, "label_vocab.json"))
    return


def download_hatespeech():
    print("downloading model for `hatespeech` classification")
    save_path = ARXIV_CHECKPOINTS["hatespeech"]
    if _create_paths(save_path):
        _download_file_from_google_drive('1BNednm_ToeNMQWGphH1Mv8xZT0Ra-Bsf', os.path.join(save_path, "model.pth.tar"))
        _download_file_from_google_drive('12XdoNIb6qdxdUYQdxjpgYjQ4gUJmefjR',
                                         os.path.join(save_path, "label_vocab.json"))
    return


def download_lid():
    print("downloading model for `lid` tagging")
    save_path = ARXIV_CHECKPOINTS["lid"]
    if _create_paths(save_path):
        _download_file_from_google_drive('190qOd6exJ8LSTmOB1VuBIARnu4drehzu', os.path.join(save_path, "model.pth.tar"))
        _download_file_from_google_drive('166DCej1fLgs8R7p-oo5E9hjY7Jko4Kth',
                                         os.path.join(save_path, "label_vocab.json"))
    return


def download_pos():
    print("downloading model for `pos` tagging")
    save_path = ARXIV_CHECKPOINTS["pos"]
    if _create_paths(save_path):
        _download_file_from_google_drive('1WjLTk2ytpDGWw4MaDMQQ5TYZVYyfF_A3', os.path.join(save_path, "model.pth.tar"))
        _download_file_from_google_drive('1_IMvERFytA2vmKWJI_7qTWmedDBOflQV',
                                         os.path.join(save_path, "label_vocab.json"))
    return


def download_ner():
    print("downloading model for `ner` tagging")
    save_path = ARXIV_CHECKPOINTS["ner"]
    if _create_paths(save_path):
        _download_file_from_google_drive('1gRUQuYhTau0bx77dxZ754oDtr-Fktnf5', os.path.join(save_path, "model.pth.tar"))
        _download_file_from_google_drive('1c3WfDr7bGYjF5t95rI-S_7jrG5Ho-Xm-',
                                         os.path.join(save_path, "label_vocab.json"))
    return


def download_mt(overwrite=False):
    print("downloading model for `machine translation`")
    save_path = ARXIV_CHECKPOINTS["mt"]
    if _create_paths(save_path) or overwrite:
        _download_file_from_google_drive('1LIGtOGU91edJ4YZ2h6x0MUt7B-FYjNDB', os.path.join(save_path, "dict.hing.txt"))
        _download_file_from_google_drive('1Y-g7GMnaqL5vEEOlNuH2V5GHkkLes6fh', os.path.join(save_path, "dict.eng.txt"))
        _download_file_from_google_drive('1NJjI8BYJh00FTIbSxcScatTVskYi8973', os.path.join(save_path, "spm8000.model"))
        _download_file_from_google_drive('1-6T-uZys8e-uMYygL9GKazjlotw4vim4', os.path.join(save_path, "spm8000.vocab"))
        _download_file_from_google_drive('1YOShDvV-ou_vfr43wlsI1xf0CZvs4Z6c',
                                         os.path.join(save_path, "checkpoint_best.pt"))
    return


def download_all():
    download_sentiment()
    download_aggression()
    download_hatespeech()
    download_lid()
    download_pos()
    download_ner()
    download_mt()
    return
