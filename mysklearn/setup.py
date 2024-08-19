# Import required functions
from setuptools import setup, find_packages

# Call setup function
setup(
    author="AP",
    description="A complete package for logistic regression.",
    name="mysklearn",
    version="0.1.0",
    packages=find_packages(include=["mysklearn","mysklearn.*"]),
    install_requires =["pandas","scipy","matplotlib","seaborn","scikit-learn"]
)
