import os

from .utils import get_module_or_attr, is_module_available


def install_all_extras():
    install_indic_trans()
    test_indic_trans()
    install_flask_extras()
    # install_fasttext()
    return


def install_indic_trans():
    if os.path.exists("indic-trans"):
        print("a folder named `indic-trans` already exists")
        return
    os.system("git clone https://github.com/libindic/indic-trans")
    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, "indic-trans"))
    os.system("pip install .")
    os.chdir(cwd)
    return


def test_indic_trans():
    if not is_module_available("indictrans"):
        raise ImportError("Install `indictrans` by running `python extras.install_indic_trans` before testing it")
    Transliterator = get_module_or_attr("indictrans", "Transliterator")
    trn = Transliterator(source='hin', target='eng', build_lookup=True)
    hin = "कांग्रेस पार्टी अध्यक्ष सोनिया गांधी, तमिलनाडु की मुख्यमंत्री"
    eng = trn.transform(hin)
    if not eng == "congress party adhyaksh sonia gandhi, tamilnadu kii mukhyamantri":
        print(eng)
        print("test_indic_trans: Fail")
    print("test_indic_trans: Success")
    return


def install_flask_extras():
    os.system("pip install flask flask_cors")
    return


def install_fasttext():
    if os.path.exists("fastText"):
        print("a folder named `fastText` already exists")
        return
    os.system("git clone https://github.com/facebookresearch/fastText")
    cwd = os.getcwd()
    nwd = os.path.join(cwd, "fastText")
    os.chdir(nwd)
    os.system("pip install .")
    os.chdir(cwd)
    os.removedirs(nwd)
    return
