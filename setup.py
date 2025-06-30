#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()


setup(
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    test_suite="vessel_analysis_3d/tests",
    version="0.0.1", 
    zip_safe=False,
)
