from types import SimpleNamespace

def dict_to_object(d: dict) -> object:
    """Convert dict instance to object instance"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_object(v) for k, v in d.items()})
    else:
        return d
