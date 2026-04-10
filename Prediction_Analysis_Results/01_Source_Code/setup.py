from setuptools import setup, find_packages

setup(
    name="petromind",
    version="0.1.0",
    description="PetroMind predictive maintenance ML pipeline",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "torch>=2.0",
        "scikit-learn>=1.3",
        "openpyxl>=3.1",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
)
