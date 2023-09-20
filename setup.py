# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

# To use a consistent encoding
from codecs import open

import setuptools

_deps = [
    "transformers==4.33",
    "accelerate==0.21",
    "einops==0.6.0",
    "omegaconf==2.3.0",
    "webdataset>=0.2.39",
    "datasets",
    "wandb",
    "sentencepiece",  # for T5 tokenizer
    "plotly",
    "pandas",
]

_extras_dev_deps = [
    "black[jupyter]~=23.1",
    "isort>=5.5.4",
    "flake8>=3.8.3",
]


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
    install_requires=_deps,
    extras_require={
        "dev": [_extras_dev_deps],
    },
)
