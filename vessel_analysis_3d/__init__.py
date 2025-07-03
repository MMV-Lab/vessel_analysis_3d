# -*- coding: utf-8 -*-

"""Top-level package for 3D Vascular Structure Analysis."""
from .processing_pipeline import Pipeline3D  # noqa: F401

__author__ = "Jianxu Chen"
__email__ = "jianxuchen.ai@gmail.com"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.1.1"


def get_module_version():
    return __version__
