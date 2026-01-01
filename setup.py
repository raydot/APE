from setuptools import setup, find_packages
setup(
    name="ape",
    version="0.3.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "openai",
        "anthropic",
        "python-dotenv",
    ],
)
