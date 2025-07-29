
from setuptools import setup, find_packages

setup(
    name="TSOTA",
    version="0.3",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "mosplot",
    ],
    description="Two-stage op-amp design automation (TSOTA)",
    author="Francisco Meireles",
    author_email="up202007382@up.pt"
)
