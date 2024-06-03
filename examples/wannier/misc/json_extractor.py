import json
import os
from typing import Tuple, Union

import numpy as np
import pandas as pd


def json_to_dict(json_file_path):
    """
    Method to load the json file to dictionary (provided a dictionary was converted to a json file )
    Arguments:
        json_file_path : str = path of json file
    Returns:
        loaded_dict : dict = dictionary associated with the json file
    """

    with open(json_file_path, "rb") as json_file:
        loaded_dict = json.load(json_file)
    return loaded_dict


def dict_to_json(dict_name, json_name):
    """
    Method to write dictionary to a json file
    Arguments:
        dict_name : str = path of dictionary to be written to json
        json_name : str = path of json file to be written
    Returns:
        none
    """
    with open(json_name, "w") as json_file:
        json.dump(dict_name, json_file)


dict_ret=json_to_dict("wannier_remsing_h2o.json")
a=list(dict_ret.keys())[6:7]
print(a)
for entries in a:
    dict1=dict_ret[entries]
    entry2=list(dict1.keys())
    print(entry2)
    for keys in entry2:
        print(entries,str(keys))
        dict2=dict1[str(keys)]
        print(entries, dict2)
        print("length of {} is {}",(entries,len(dict2)))
