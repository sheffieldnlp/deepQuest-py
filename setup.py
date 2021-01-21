from setuptools import find_packages, setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="deepquestpy",
    version="0.1.0",
    author="SheffieldNLP",
    description="Pytorch version of DeepQuest - a library for Quality Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sheffieldnlp/deepQuest-py",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=["numpy", "tqdm", "transformers", "scipy", "scikit-learn", "pandas", "tokenizers", "datasets"],
    entry_points={"console_scripts": ["deepquestpy-run-model=deepquestpy_cli.run_model:main"]},
)
