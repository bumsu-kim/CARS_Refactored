from setuptools import setup, find_packages

setup(
    name="cars",
    version="1.0",
    packages=find_packages(),
    description="CARS: Curvature-Aware Random Search method for derivative-free optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Bumsu Kim, Daniel McKenzie, HanQin Cai, and Wotao Yin",
    author_email="bumsu@ucla.edu",
    url="https://github.com/bumsu/CARS_Refactored",
    install_requires=[
        "numpy",
        # etc.
    ],
    python_requires=">=3.6",
    license="MIT",
)
