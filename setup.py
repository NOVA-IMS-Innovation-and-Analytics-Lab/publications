from setuptools import setup, find_packages

setup(
    name="tools",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'requests>=2.0.0',
        'xlrd>=1.0.0',
        'cluster-over-sampling>=0.1.0',
        'geometric-smote>=0.1.2',
        'research-learn == 0.1.0'
    ],
    entry_points={
        'console_scripts':['run=tools.cli:run']
    }
)
