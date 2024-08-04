from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="lutnn",
    version="0.1.0",
    author="Iván Ramírez Díaz",
    author_email="ivan.ramirez@urjc.es",
    description="LUTNN Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/capo-urjc/lutnn",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        # Add your project's dependencies here
        # e.g. "numpy>=1.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            # Other development dependencies
        ],
    },
    include_package_data=True,
)
