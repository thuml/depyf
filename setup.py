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
    ],
    extras_require={
        "dev": [
            "pytest",
        ]
    }
)