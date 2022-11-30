import json
from pathlib import Path


def update_json(json_path:Path, items: dict) -> dict:
    """Updates JSON in json file, creates new file if non-existent.

    Args:
        json_path (Path): Full path to file
        items (dict): To be updated JSON

    Returns:
        dict: _description_
    """
    try:
        if not json_path.is_file():
            with open(json_path, 'w') as file:
                json.dump({}, file)

        with open(json_path, 'r') as file:
            json_data = json.load(file)

        json_data.update(items)

        with open(json_path, 'w') as file:
            json.dump(json_data, file, indent=4)
    except:
        with open(json_path, 'w') as file:
            json.dump(json_data, file, indent=4)

    return json_data


def load_json(json_path:Path) -> dict:
    """Load the content of a JSON file.

    Args:
        json_path (Path): Full path to file

    Returns:
        dict: JSON content
    """
    if not json_path.is_file():
        return {}

    with open(json_path, 'r') as file:
        json_data = json.load(file)

    return json_data