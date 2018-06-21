
from setuptools import setup

setup(name = 'sdev_pycore_utils',
      version ='0.1',
      description='Utility functions for dealing with python std lib functionality',
      url='',
      author='Andres_mariscal',
      author_email='carlos.mariscal.melgar@gmail.com',
      license='MIT',
      packages=['sdev_pycore_utils'],
      install_requires=[
          'dill',
          'comtypes',
          ],
      zip_safe=False)
