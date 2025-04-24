from setuptools import setup, find_packages

setup(
    name="sap_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.36.0",
        "python-dotenv>=1.0.0",
        "pdfplumber>=0.10.3",
        "torch>=2.1.0",
        "tqdm>=4.66.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
    python_requires=">=3.8",
) 