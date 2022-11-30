import re
import os
import pytest
import pdb
from pathlib import Path


@pytest.fixture(scope="session")
def root_folder(pytestconfig):
    return pytestconfig.getoption("root_folder")


def test_main(root_folder: str) -> None:
    """This test checks weather the branch name respects our naming conventions.

    Args:
        root_folder (str): root folder of the git repository
    """

    locations = ["data", "etc", "notebooks", "local_runner", "src"]
    exclude_locations = ["__pycache__"]
    regex = "^[a-z_.]+$"
    match_array = []
    fs_entities = []

    # ------------------------------------ Act ----------------------------------- #

    for location in locations:
        path = Path(root_folder, location)

        for fs_entity in path.glob("**/*"):
            print(fs_entity)
            for ex_loc in exclude_locations:
                if ex_loc in str(fs_entity.parent) or fs_entity.name.startswith(ex_loc):
                    break
            else:
                match_array.append(re.search(regex, str(fs_entity.name)))
                fs_entities.append(fs_entity)

    # ---------------------------------- Assert ---------------------------------- #

    for match_object, fs_entity in zip(match_array, fs_entities):
        assert match_object is not None, f"File system entity name '{fs_entity.name}' at '{fs_entity.parent}' does not conform to naming conventions"


if __name__ == "__main__":
    test_main()