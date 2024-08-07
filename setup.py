from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

    setup(
    name='torch_canon',
    version='0.1',
    description='PyTorch-Canon',
    author='Justin Baker',
    author_email='baker@math.utah.edu',
    packages=['torch_canon'],  #Package Name
    )
