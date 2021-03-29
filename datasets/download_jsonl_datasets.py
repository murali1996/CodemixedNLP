import argparse
import requests
import tarfile
import os
import shutil


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def cleanse_folder(directory, prefix):
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            cleanse_folder(path, prefix)
        if item.startswith(prefix):
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(os.path.join(directory, item)):
                shutil.rmtree(path)
            else:
                print("A simlink or something called {} was not deleted.".format(item))
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="data downloading")

    curdir = os.path.abspath("./")
    dataset_folder = "../datasets"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    os.chdir(dataset_folder)

    file_urls = {
        "semantic_jsonl_datasets.tar.gz": ["16WY9JLGh5OiTUtPF-FHbmRGyaOhrsxon"],
        "syntactic_jsonl_datasets.tar.gz": ["1W3LoeOOFEkHckCPy8_cnD2F6YCIyi5SA"]
    }

    for destination, file_id in file_urls.items():

        for file_id_e in file_id:
            download_file_from_google_drive(file_id_e, destination)
            tar = tarfile.open(destination, "r:gz")
            tar.extractall()
            tar.close()
            os.remove(destination)

    cleanse_folder(os.path.join(curdir, dataset_folder), "._")

    os.chdir(curdir)
