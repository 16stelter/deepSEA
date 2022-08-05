from setuptools import setup

package_name = 'deep_sea'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sebi',
    maintainer_email='6stelter@informatik.uni-hamburg.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'deep_sea = deep_sea.deepsea_pywrapper:main',
        ],
    },
)
