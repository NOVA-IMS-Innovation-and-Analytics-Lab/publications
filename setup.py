from setuptools import setup, find_packages

setup(
    name="tools",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'sklearnext @ https://github.com/IMS-ML-Lab/scikit-learn-extensions'
    ],
    entry_points={
        'console_scripts':['run=tools.cli:run']
    }
)
