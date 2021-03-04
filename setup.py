#!/usr/bin/env python3
import os
from setuptools import find_packages
from setuptools import setup

requirements = {
    "install": [
        "kaldi_io",
        "opencv-python==3.4.*",
        "opencv-contrib-python==3.4.*",
        "matplotlib"
    ],
}
install_requires = requirements["install"]
dirname = os.path.dirname(__file__)
setup(name='hynet',
      version='0.1.0',
      author='jpong',
      author_email='ljh93ljh@gmail.com',
      description='hynet',
      license='Apache Software License',
      packages=find_packages(include=['hynet*']),
      install_requires=install_requires,
      classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Topic :: Software Development :: Libraries :: Python Modules'],
      )
