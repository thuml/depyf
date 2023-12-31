from setuptools import setup, find_packages

setup(
    name='depyf',
    version=open("depyf/VERSION.txt").read().strip(),
    description='Decompile python functions, from bytecode to source code!',
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url='https://github.com/thuml/depyf',
    author='Kaichao You',
    author_email="youkaichao@gmail.com",
    license="MIT",
    include_package_data=True,  # This line is important!
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "astor",
        "dill", # used for serializing code object. PyTorch bytecode is not serializable by marshal. Check https://github.com/pytorch/pytorch/issues/116013 for details.
        # "filelock", # filelock is required by torch. If you use torch, you should have filelock.
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
            "autopep8",
        ]
    }
)