import os


def fix_filename(filename):
    return os.path.join("config", filename) if filename[:6] != "config" else filename
