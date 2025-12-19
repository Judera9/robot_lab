"""Installation script for the 'robot_lab' python package."""

from setuptools import setup, find_packages

# Installation operation
setup(
    name="robot_lab",   
    package_dir={"": "source/robot_lab"},
    packages=find_packages(where="source/robot_lab"),
    description="Robot Lab for Humanoid Motion, Modified by Jude",
    author="Ziqi Fan, Jude",
    url="https://github.com/your-repo/robot_lab",
    version="2.3.0",
    install_requires=[
        "numpy==1.26.0",
        # "packaging==23.0",
    ],
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)
