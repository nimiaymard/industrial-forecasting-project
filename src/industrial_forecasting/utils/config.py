import yaml
from types import SimpleNamespace

def load_config(path="config.yaml"):
    with open(path, "r") as file:
        cfg = yaml.safe_load(file)

    # Convertit les dictionnaires imbriqués en objets accessibles par dot notation
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d

    return dict_to_namespace(cfg)
