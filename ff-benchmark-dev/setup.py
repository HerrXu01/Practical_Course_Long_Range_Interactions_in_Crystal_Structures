import setuptools

with open("models/matformer/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="matformer",
    version="2022.09.15",
    author="Keqiang Yan, Yi Liu, Yuchao Lin, Shuiwang Ji",
    author_email="keqiangyan@tamu.edu",
    description="matformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YKQ98",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
