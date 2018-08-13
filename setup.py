# !/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='moajoloader',
    version='0.1',
    description='data loader for ML',
    author='moajo',
    author_email='mimirosiasd@gmail.com',
    url='none',
    packages=find_packages(),
    entry_points="""
    [console_scripts]
    """,
    install_requires=['tqdm'],
    include_package_data=True,
)
