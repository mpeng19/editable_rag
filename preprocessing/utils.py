"""
This module contains utility functions for preprocessing.
"""
import json

def save_json(data, filename):
    """
    Save data to a JSON file.

    Args:
        data (any): Data to save.
        filename (str): Name of the file to save the data.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(filename):
    """
    Load data from a JSON file.

    Args:
        filename (str): Name of the file to load the data.
    """
    with open(filename, encoding = 'utf-8') as f:
        data = json.load(f)
    return data