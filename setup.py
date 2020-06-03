import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='aligned_bert_embedder',
    version='0.7',
    scripts=['aligned_bert_embedder'],
    authors=["Adaxry", "Zeio Nara"],
    author_email="zeionara@gmail.com",
    description="Module for generating aligned contextualized bert embeddings using different strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zeionara/aligned_bert_embedder",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ]
)
