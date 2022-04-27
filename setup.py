from setuptools import setup, find_packages

setup(
  name='SBI-Diff-Simulator',
  version='0.0.1',
  url='',
  author='',
  description='',
  packages=find_packages(),
  install_requires=[
    'numpy>=1.19.2',
    'jax>=0.2.0',
    'tensorflow_probability>=0.14.1',
    'scikit-learn>=0.21',
    'jaxopt>=0.2'
  ],
)