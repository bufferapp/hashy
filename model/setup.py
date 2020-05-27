#!/usr/bin/env python

"""Scikit-learn trainer package setup."""

import setuptools


REQUIRED_PACKAGES = [
    "scikit-learn==0.20.4",
    "pandas==0.24.2",
    "pandas-gbq==0.13.2",
    "cloudml-hypertune",
    "gensim==3.8.3",
]

setuptools.setup(
    name="hashy",
    version="0.1.0",
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
    description="Hashtag Suggestions",
)
