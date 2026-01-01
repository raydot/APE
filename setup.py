from setuptools import setup, find_packages
setup(
    name="ape",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "openai",
        "anthropic",
        "python-dotenv",
    ],
)
