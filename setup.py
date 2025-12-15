from setuptools import setup, find_packages

setup(
    name="182-final-project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
        "numpy",
        "torch",
        "Pillow",
    ],
)