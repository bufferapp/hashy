#!/usr/bin/env python

"""Scikit-learn trainer package setup."""

import setuptools


REQUIRED_PACKAGES = [
    "scikit-learn",
    "pandas-gbq",
    "cloudml-hypertune",
    "gensim",
]

setuptools.setup(
    name="hashy",
    version="0.1.0",
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
    description="Hashtag Suggestions",
)
