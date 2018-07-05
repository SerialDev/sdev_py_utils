
from setuptools import setup

setup(name = 'sdev_lib_utils',
      version ='0.1',
      description='Utility functions for dealing with non std libraries',
      url='',
      author='Andres_mariscal',
      author_email='carlos.mariscal.melgar@gmail.com',
      license='MIT',
      packages=['sdev_lib_utils'],
      install_requires=[
          'boto3',
          ],
      zip_safe=False)
        
