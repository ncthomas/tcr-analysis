from setuptools import setup, find_packages

setup(
    name='TCR Analysis',
    author='Niclas Thomas',
    version='0.1.0',
    packages=find_packages(),
    long_description=open('README.md').read(),
    install_requires=[
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'networkx',
        'python-Levenshtein'
    ]
)