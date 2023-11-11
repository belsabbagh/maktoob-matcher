import datetime
import os


def datetime_to_float(d):
    epoch = datetime.datetime.utcfromtimestamp(0)
    total_seconds = (d - epoch).total_seconds()
    return total_seconds


def files_iter():
    for f in os.listdir("data/processed"):
        if not f.startswith("data"):
            continue
        yield f
