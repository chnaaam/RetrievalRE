import os


def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def rm_dir(path):
    if os.path.isdir(path):
        os.rmdir(path)
    