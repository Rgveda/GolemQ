# coding:utf8
from .api import *
from .helpers import get_stock_codes, update_stock_codes
from astartool.setuptool import get_version

VERSION = (0, 8, 0, 'final', 0)

__version__ = get_version(VERSION)
__author__ = "A.Star"

del get_version
