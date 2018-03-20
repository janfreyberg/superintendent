"""Installer."""
from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

blurb = ''
if path.isfile('README.md'):
    readme = open('README.md', 'r').read()
else:
    readme = blurb

version = '0.0.1'

setup(
    name='superintendent',
    version=version,
    description=blurb,
    long_description=readme,
    url='https://github.com/janfreyberg/superintendent',
    download_url='https://github.com/janfreyberg/superintendent/' +
        version + '.tar.gz',
    # Author details
    author='Jan Freyberg',
    author_email='jan.freyberg@gmail.com',
    packages=['superintendent'],
    keywords=['widgets', 'labelling', 'annotation'],
    install_requires=['ipywidgets', 'ipyevents', 'numpy',
                      'pandas', 'matplotlib']
    # Include the template file
    # package_data={
    #     '': ['data/*nii*',
    #          'data/examples_surfaces/lh.*',
    #          'data/examples_surfaces/*.ctab']
    # },
)
