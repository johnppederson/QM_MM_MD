"""
setup.py: Builds repository and distribution information.
"""
import setuptools

__author__ = "Shahriar Khan and John Pederson"
__version__ = "0.0.1"

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(name="QM_MM_MD",
                 version="0.0.1",
                 description="Extension of QM_MM to model QM/MM dynamics",
                 author="Shahriar Khan and John Pederson",
                 author_email="jpederson6@gatech.edu",
                 packages=["qm_mm_md"],
                 python_requires=">3.5")
