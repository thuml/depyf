from setuptools import setup, find_packages

setup(
    name='depyf',
    version='0.1.0',
    description='Decompile python functions, from bytecode to source code!',
    long_description=open("README.md", "r").read(),
    url='https://github.com/youkaichao/depyf',
    author='Kaichao You',
    author_email="youkaichao@gmail.com",
    license="MIT",
    packages=find_packages(include=["depyf"]),
    python_requires=">=3.7",
    install_requires=[
        "astor",
    ],
    extras_require={
        "dev": [
            "pytest",
            "networkx",
            "matplotlib"
        ]
    }
)