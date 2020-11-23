# coding=utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2020 azai/GolemQ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
try:
    from setuptools import setup, find_packages
except:
    from distutils.core import setup
"""
"""

if sys.version_info.major != 3 or sys.version_info.minor not in [5, 6, 7, 8]:
    print('wrong version, should be 3.5/3.6/3.7/3.8 version')
    sys.exit()

# or
# from distutils.core import setup  

setup(
        name='GolemQ',     # 包名字
        version='0.0.2',   # 包版本
        description='A PowerToys based QUANTAXIS',   # 简单描述
        author='azai',  # 作者
        author_email='11652964@qq.com',  # 作者邮箱
        url='https://github.com/Rgveda/GolemQ',      # 包的主页
        packages=find_packages(),                # 包
)

