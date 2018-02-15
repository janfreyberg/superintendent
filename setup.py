"""Installer."""
from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

blurb = 'A package that provides ipywidgets for standard neuroimaging plotting'
if path.isfile('README.md'):
    readme = open('README.md', 'r').read()
else:
    readme = blurb

version = '0.0.1'

setup(
    name='niwidgets',
    version=version,
    description=blurb,
    long_description=readme,
    url='https://github.com/janfreyberg/sueprintendent',
    download_url='https://github.com/janfreyberg/sueprintendent/' +
        version + '.tar.gz',
    # Author details
    author='Jan Freyberg',
    author_email='jan.freyberg@gmail.com',
    packages=['superintendent'],
    keywords=['widgets', 'labelling', 'annotation'],
    install_requires=['ipywidgets']
    # Include the template file
    # package_data={
    #     '': ['data/*nii*',
    #          'data/examples_surfaces/lh.*',
    #          'data/examples_surfaces/*.ctab']
    # },
)
