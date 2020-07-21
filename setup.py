from pathlib import Path
from setuptools import find_packages, setup

setup(
    name="Brill",
    version=1.0,
    description="Code based on Brill from A.Gaup and M.Scheer",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Michael Scheer et al.",
    license="GNU General Public License v3.0",
    pymodules=["brill.py"],
    install_requires=["numpy", "scipy", "matplotlib"],
    python_requires=">=3.6",
    entry_points={"console_scripts": ["brill=brill:main"]},
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
