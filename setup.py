from setuptools import setup, find_packages
from typing import Optional

import subprocess
import sys

def get_git_commit_id(n_digits=8) -> Optional[str]:
    try:
        # Run the git command to get the current commit ID
        commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        # Decode bytes to string (Python 3.x)
        commit_id = commit_id.decode('utf-8')
        return commit_id[:n_digits]
    except subprocess.CalledProcessError as e:
        print("Failed to get Git commit ID:", e)
        return None

def get_version():
    # Read the version from the VERSION.txt file
    with open("depyf/VERSION.txt", "r") as f:
        version = f.read().strip()
    # If the current commit ID is available, append it to the version
    commit_id = get_git_commit_id()
    # do not append commit_id if we are building sdist wheels for release
    if commit_id and "sdist" not in sys.argv:
        version += "+" + commit_id
    return version

setup(
    name='depyf',
    version=get_version(),
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