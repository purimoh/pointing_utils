# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['pointing_utils', 'pointing_utils.optimal_control']

package_data = \
{'': ['*']}

install_requires = \
['control>=0.9.3,<0.10.0',
 'emgregs>=0.0.4,<0.0.5',
 'matplotlib>=3.7.0,<4.0.0',
 'numpy>=1.24.2,<2.0.0',
 'pandas>=1.5.3,<2.0.0',
 'statsmodels>=0.13.5,<0.14.0',
 'tabulate>=0.9.0,<0.10.0']

setup_kwargs = {
    'name': 'pointing-utils',
    'version': '0.0.2.dev0',
    'description': 'Utilities for dealing with pointing data',
    'long_description': None,
    'author': 'jgori',
    'author_email': 'juliengori@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<3.11',
}


setup(**setup_kwargs)
