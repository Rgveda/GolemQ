# coding:utf8

from astartool.error import ParameterValueError
from . import boc, daykline, hkquote, jsl, sina, tencent, timekline

USE_MAP = {
    'sina': sina.Sina,
    'jsl': jsl.Jsl,
    'qq': tencent.Tencent,
    'tencent': tencent.Tencent,
    'boc': boc.Boc,
    'timekline': timekline.TimeKline,
    'daykline': daykline.DayKline,
    'hkquote': hkquote.HKQuote
}


# pylint: disable=too-many-return-statements
def use(source):
    if source not in USE_MAP:
        raise ParameterValueError("没有找到对应的接口")
    else:
        return USE_MAP[source]()
