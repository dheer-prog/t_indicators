from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        't_indicators',
        ['binding.cpp', 'SMA.cpp', 'RSI.cpp', 'williams_r.cpp', 'trend.cpp', 'volatility.cpp', 'probability.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++17'],
    ),
]

setup(
    name='t_indicators',
    version='0.1',
    ext_modules=ext_modules,
)
#python setup.py build_ext --inplace (how to build)