from os import path

from setuptools import setup, find_packages

setup(
    name="ma_meta_env",
    version="0.0.1",
    url="https://github.com/hzyjerry/ma-meta-env",
    py_modules=["ma_meta_env"],
    packages=find_packages(),
    author="Zhiyang He",
    author_email="hzyjerry@berkeley.com",
    install_requires=["matplotlib", "seaborn"],
    tests_require=["pytest"],
    python_requires=">=3.6",
)
