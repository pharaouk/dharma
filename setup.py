import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dharma",
    version="1.0.0",
    author="pharaouk",
    author_email="pharaouk@gmail.com",
    description="dharma",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/pharaouk/dharma',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'dharma=dharma:run_dharma',
        ],
    },
    package_data={
        'dharma': ['benchmarks/*.jsonl'],
    },
    install_requires=['numpy','scipy','requests', 'datasets', 'pyyaml'],
) 
