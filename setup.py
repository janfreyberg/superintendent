"""Installer."""
from setuptools import setup
# To use a consistent encoding
from codecs import open
import os.path

here = os.path.dirname(os.path.abspath(__file__))

version_ns = {}
with open(os.path.join(here, 'superintendent', 'version.py')) as f:
    exec(f.read(), {}, version_ns)

version = version_ns['version']

blurb = ''
if os.path.isfile('README.md'):
    readme = open('README.md', 'r').read()
else:
    readme = blurb

setup(
    name='superintendent',
    version=version,
    description=blurb,
    long_description=readme,
    url='https://github.com/janfreyberg/superintendent',
    download_url='https://github.com/janfreyberg/superintendent/' +
        version_ns['version'] + '.tar.gz',
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
