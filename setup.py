from setuptools import setup, find_packages

setup(
    name='symspellpy',
    packages=find_packages(exclude=['test']),
    package_data={
        'symspellpy': ['README.md', 'LICENSE']
    },
    version='0.9.0',
    description='Keyboard layout aware version of SymSpell',
    long_description=open('README').read(),
    author='crossnox',
    url='https://github.com/crossnox/symspellpy',
    keywords=['symspellpy'],
    install_requires=[
        'scipy >= 0.19'
    ],
    python_requires='>=3.4',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'License :: OSI Approved :: MIT License',
    ],
    test_suite="test",
    entry_points={

    }
)
