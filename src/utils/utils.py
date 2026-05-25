"""Utility functions for the project. These functions provide common operations that are highly generic."""

import datetime as dt
import os
import re


def create_datetime_id():
    """
    Creates a unique identifier based on the current datetime.

    :return: unique identifier
    :rtype: str
    """
    now = dt.datetime.now()
    dt_id = now.strftime("%Y%m%d%H%M%S")
    return dt_id


def make_dir(path):
    """
    Creates a directory if it does not already exist.

    :param pathlib.Path path: path of new directory to create
    :return: None
    :rtype: None
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def camel_to_snake(string):
    """
    Converts a string from camel to snake case.

    :param str string: input string to be converted to snake case
    :return: the given string in snake case
    :rtype: str
    """
    if not string:
        return string
    string = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    string = re.sub('(.)([0-9]+)', r'\1_\2', string)
    string = re.sub('([a-z0-9])([A-Z])', r'\1_\2', string).lower()
    return string
