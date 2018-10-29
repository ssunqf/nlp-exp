#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='local_attention',
    ext_modules=[
        CppExtension('lltm_cpp', ['lltm.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })