from setuptools import find_packages, setup

setup(
    name="deepquestpy",
    version="0.1.0",
    description="deepQuest-py: Implementing and Evaluating Quality Estimation Models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="deepquest deepquestpy NLP quality estimation machine translation",
    url="https://github.com/sheffieldnlp/deepQuest-py",
    author="SheffieldNLP",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "allennlp==2.1.0",
        "numpy",
        "tqdm",
        "transformers",
        "scipy",
        "scikit-learn",
        "pandas",
        "tokenizers",
        "datasets",
    ],
    entry_points={"console_scripts": ["deepquestpy-run-model=deepquestpy_cli.run_model:main"]},
)
