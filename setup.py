from setuptools import setup, find_packages

setup(
  name='sbids',
  version='0.0.1',
  url='https://github.com/Justinezgh/SBI-Diff-Simulator',
  description='SBI Diff Simulator',
  packages=find_packages(),
  package_dir={'sbids': 'sbids'},
  package_data={
      'sbids': ['data/*.csv', 'data/*.npy', 'data/*.pkl'],
   },
  include_package_data=True,
  install_requires=[
    'numpy>=1.19.2',
    'jax>=0.2.0',
    'tensorflow_probability>=0.14.1',
    'scikit-learn>=0.21',
    'dm-haiku==0.0.5',
    'jaxopt>=0.2',
    'numpyro==0.10.1',
    'jax-cosmo>=0.1.0'
  ],
)
