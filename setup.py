from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="elsa",
    version="0.1.0",
    author="Claude & Jacob",
    description="Embedding Locus Shingle Alignment - syntenic block discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "scikit-learn",
        "numpy",
        "pandas",
        "pyarrow",
        "biopython",
        "click",
        "rich",
        "pydantic>=2.0",
        "pyyaml",
        "datasketch",
        "fair-esm",
        "fastapi",
        "uvicorn",
    ],
    entry_points={
        "console_scripts": [
            "elsa=elsa.cli:main",
            "micro-synteny=elsa.synteny.micro_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3.10",
    ],
)
