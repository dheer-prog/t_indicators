import os
from setuptools import setup, Extension
import pybind11

if os.name == 'nt':
    compile_args = ['/O2', '/std:c++17', '/DNDEBUG']
    link_args = []
else:
    compile_args = ['-std=c++17', '-O3', '-DNDEBUG', '-march=native', '-flto']
    link_args = ['-flto','-pthread']

ext_modules = [
    Extension(
        't_indicators',
        ['src/binding.cpp', 'src/SMA.cpp', 'src/RSI.cpp', 'src/williams_r.cpp', 'src/trend.cpp', 'src/volatility.cpp', 'src/probability.cpp', 'src/ema.cpp', 'src/macd.cpp', 'src/csv_loader.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
]

setup(
    name='t_indicators',
    version='0.2',
    ext_modules=ext_modules,
)
#You need to add intraday function into williams_r"
#Also need to fix trends.cpp
#Also need to add multi-threading to RSI.cpp
"""
Things which are completed and don't touch
SMA 
EMA
RSI
WILLIAMS_R
"""
#python setup.py build_ext --inplace (how to build)
