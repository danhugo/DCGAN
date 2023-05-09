from types import SimpleNamespace

def dict_to_object(d):
    """Convert dict to Namespace object"""
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_object(v)
    return SimpleNamespace(**d)

            