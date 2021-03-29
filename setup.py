from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="CodemixedNLP",
    version="0.0.1",
    author="Sai Muralidhar Jayanthi, Kavya Nerella",
    author_email="jsaimurali001@gmail.com",
    description="CodemixedNLP: An Extensible and Open Toolkit for Code-Switched NLP",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",
    url="",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">3.6",
    install_requires=[
        'torch==1.7.0',
        'sklearn',
        'numpy',
        'tqdm',
        'jsonlines',
        'wordsegment',
        'pytorch_pretrained_bert',
        'transformers~=3.5.1',
        'matplotlib',
    ],
    keywords="transformer networks Hinglish NLP embedding PyTorch Hindi English code switch code-mix deep learning"
)
