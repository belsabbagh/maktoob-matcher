"""This module is responsible for fetching data from the dataset source and storing it in the `data/raw` folder."""
import os
import zipfile
import wget


def download_and_extract(url, out):
    filepath = wget.download(url, out="data/tmp")
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(out)
    # clear tmp folder
    os.remove(filepath)


download_fns = [download_and_extract]

urls = {"https://fada.birzeit.edu/bitstream/20.500.11889/6023/1/AninisAuthorshipAttributionDataset.zip": 0}

if __name__ == "__main__":
    for url in urls:
        download_fns[urls[url]](url, "data/raw")
