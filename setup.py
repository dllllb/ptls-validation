import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ptls-validation",
    version="0.0.3",
    author="",
    author_email="",
    description="Estimate your feature vector quality on downstream task",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dllllb/plts-valid",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'luigi>=3.0.0',
    ],
)
