# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2016-2018 yutiansut/QUANTAXIS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Persistence package — data loading and completeness checks.

Split into focused sub-modules (each ≤300 lines):
- _schema:   Column definitions and helper functions
- _review:   Stock review persistence check (hourly features)
- _concept:  Concept and massive model persistence checks
- _stock:    Multi-timeframe stock reality feature persistence check
- _daily:    Daily stock reality feature persistence check
"""

from GolemQ.services.persistence._schema import (
    concept_review_columns_of_persistence,
    stock_review_columns_of_persistence,
    massive_review_columns_of_persistence,
    reality_columns_of_persistence,
    daily_columns_of_persistence,
    calc_masked_tail_missing_index,
    features_reasonableness_checks,
)

from GolemQ.services.persistence._review import (
    dataloader_review_check_reflush,
    dataloader_review_check,
)

from GolemQ.services.persistence._concept import (
    dataloader_concept_check,
    dataloader_massive_check,
)

from GolemQ.services.persistence._stock import (
    dataloader_persistence_check,
)

from GolemQ.services.persistence._daily import (
    dataloader_persistence_daily_check,
)

__all__ = [
    # Schema
    'concept_review_columns_of_persistence',
    'stock_review_columns_of_persistence',
    'massive_review_columns_of_persistence',
    'reality_columns_of_persistence',
    'daily_columns_of_persistence',
    'calc_masked_tail_missing_index',
    'features_reasonableness_checks',
    # Review
    'dataloader_review_check_reflush',
    'dataloader_review_check',
    # Concept
    'dataloader_concept_check',
    'dataloader_massive_check',
    # Stock
    'dataloader_persistence_check',
    # Daily
    'dataloader_persistence_daily_check',
]
