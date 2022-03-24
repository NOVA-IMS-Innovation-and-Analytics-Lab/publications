#! /usr/bin/env python
"""Code to reproduce various publications."""

from setuptools import find_packages, setup

DISTNAME = 'publications'
DESCRIPTION = 'Code to reproduce various publications.'
MAINTAINER = 'G. Douzas'
MAINTAINER_EMAIL = 'gdouzas@icloud.com'
URL = 'https://github.com/NOVA-IMS-Innovation-and-Analytics-Lab//publications'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/NOVA-IMS-Innovation-and-Analytics-Lab//publications'
CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved',
    'Programming Language :: Python',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
]
INSTALL_REQUIRES = [
    'mlflow >= 1.0.0',
    'pandas >= 1.2.0',
    'rich >= 10.0.0',
    'requests >= 2.26.0',
    'scikit-learn >= 1.0.0',
    'imbalanced-learn >= 0.7.0',
    'xlrd >= 2.0.0',
]
VERSION = '0.0.1'

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    download_url=DOWNLOAD_URL,
    zip_safe=False,
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    version=VERSION,
    install_requires=INSTALL_REQUIRES,
    entry_points={'console_scripts': ['run=tools.cli:run', 'list=tools.cli:list']},
)
