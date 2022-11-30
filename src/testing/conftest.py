import pytest


def pytest_addoption(parser):
    parser.addoption("--branch_name", action="store", default="fail")
    parser.addoption("--root_folder", action="store", default="fail")