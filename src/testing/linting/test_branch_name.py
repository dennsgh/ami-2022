import re
import pytest


@pytest.fixture(scope="session")
def branch_name(pytestconfig):
    return pytestconfig.getoption("branch_name")


def test_main(branch_name: str) -> None:
    """This test checks weather the branch name respects our naming conventions.

    Args:
        branch_name (str): branch name passed as git variable.
    """

    regex = "(^develop$)|"
    regex += "(^master$)|"
    regex += "(^feature_[a-zA-Z0-9_]+$)|"
    regex += "(^experimental_[a-zA-Z0-9_]+$)|"
    regex += "(^fix_[a-zA-Z0-9_]+$)|"
    regex += "(^docker_[a-zA-Z0-9_]+$)"

    # ------------------------------------ Act ----------------------------------- #

    match_object = re.search(regex, branch_name)

    # ---------------------------------- Assert ---------------------------------- #

    assert match_object is not None, f"Your branch name {branch_name} does not conform with the naming conventions"


if __name__ == "__main__":
    test_main()
