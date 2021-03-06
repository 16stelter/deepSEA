import glob

from setuptools import setup
from setuptools import find_packages


package_name = 'data_reader'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
         glob.glob('launch/*.py')),
        ('share/' + package_name + '/scripts',
         glob.glob('scripts/*.py')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sebi',
    maintainer_email='6stelter@informatik.uni-hamburg.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    scripts=['scripts/record.py'],
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'reader = data_reader.reader:main'
        ],
    },
)
