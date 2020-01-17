import importlib


def load_config(path):
    spec = importlib.util.spec_from_file_location("cfg", path)
    cfg = spec.loader.load_module()
    return cfg
