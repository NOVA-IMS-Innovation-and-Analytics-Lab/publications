from setuptools import setup, find_packages

setup(
    name="tools",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'sklearnext @ git+http://git@github.com/IMS-ML-Lab/scikit-learn-extensions.git#egg=sklearnext'
    ],
    entry_points={
        'console_scripts':['run=tools.cli:run']
    }
)
