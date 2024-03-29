#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = open("requirements.txt").read().splitlines()

test_requirements = [ ]

setup(
    author="Mehdi Cherti",
    author_email='mehdicherti@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="VQGAN from LDM without hell of dependencies",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='vqgan_nodep',
    name='vqgan_nodep',
    packages=find_packages(include=['vqgan_nodep']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/mehdidc/vqgan_nodep',
    version='0.1.0',
    zip_safe=False,
)
