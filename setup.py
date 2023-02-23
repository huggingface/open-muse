import os

# To use a consistent encoding
from codecs import open

import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# read version
with open(os.path.join(here, "muse", "__init__.py"), encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"')
            break
    else:
        raise RuntimeError("Unable to find version string.")

setuptools.setup(
    name="muse",
    version=version,
    description="The best generative model in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=["transformers", "accelerate", "einops", "omegaconf"],
    extras_require={
        "dev": ["black[jupyter]", "isort", "flake8>=3.8.3"],
    },
)
