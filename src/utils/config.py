import yaml
from typing import Union
from .instance import dict_to_object
def read_config(path, return_type: str='object', show: bool = True) -> Union[type, dict]:
    """read configuration from .yaml file 

    Arguments
    - path (str): file path
    - return_type (str): 'object' return class else return dict.
    - show (bool): True if show config,  False not to show.
    """
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    show_config(config)
    if return_type == 'object':
        return dict_to_object(config)
    else:
        return config
    
def show_config(config: dict):
    for key, value in config.items():
        print(key, value)
    print()

