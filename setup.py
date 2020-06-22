'''
Created on 12 April 2017
@author: Sambit Giri
Setup script
'''

from setuptools import setup, find_packages
#from distutils.core import setup


setup(name='emulate21cm',
      version='0.0.1',
      author='Sambit Giri',
      author_email='sambit.giri@gmail.com',
      package_dir = {'emulate21cm' : 'src'},
      packages=['emulate21cm'],
      package_data={'share':['*'],},
      install_requires=['numpy','scipy','scikit-learn','scikit-image'],
      #include_package_data=True,
)
