# coding:utf-8
"""
Risk model constants (stub — to be populated).

Used by services/persistence/ for CVaR risk feature column references.
"""


class _StubMeta(type):
    """未定义的常量一律 **抛 AttributeError**。

    此前它返回 ``'rsk_' + name.lower()`` —— 一个「看起来合理」的假名。
    后果见 ``PITFALLS.md`` P1：假名在数据里从不存在，取值静默失败，
    回测「跑通了」却零成交，**没有任何报错**。

    ⚠️ **不要改回返回值。** 若某常量抛 AttributeError，正解是从老树取
    真实值补上 —— 那才是该字段在 MongoDB 里的实际存储名。
    """
    def __getattr__(cls, name):
        raise AttributeError(
            f'{cls.__name__}.{name} 未定义。本类不再伪造常量名'
            f'（见 PITFALLS.md P1）。请从老树取真实值补上 ——'
            f'那是数据在 MongoDB 里的实际字段名。')


class RSK(metaclass=_StubMeta):
    """Risk model feature constants (stub)."""
    _instance = None

    CVaR_PEAK_PRICE = 'ES_PEAK_P'
    CVaR_PEAK_LOW = 'ES_PEAK_LO'
    CVaR_PEAK_LOW_PRICE = 'ES_PEAK_LO_P'
    CVaR_PEAK_LOW_BEFORE = 'ES_PEAK_LO_BF'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RSK, cls).__new__(cls)
        return cls._instance
