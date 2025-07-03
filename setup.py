#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import find_packages, setup

setup(
    packages=find_packages(include=["vessel_analysis_3d", "vessel_analysis_3d.*"]),
    package_dir={"vessel_analysis_3d": "vessel_analysis_3d"},
)