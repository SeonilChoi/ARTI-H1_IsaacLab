# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'arti_h1' python package."""

import os

import toml
from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # NOTE: Add dependencies
    "psutil",
]

# Installation operation
setup(
    name="arti_h1",
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    package_data={
        "arti_h1": [
            "assets/meshes/arti_h1/*.STL",
            "assets/models/arti_h1/*.urdf",
            "assets/models/arti_h1/arti_h1/*.usd",
            "assets/models/arti_h1/arti_h1/configuration/*.usd",
            "assets/motions/arti_h1/*.json",
        ]
    },
    # `isaaclab train|play|list_envs` import every package registered under this group (isaaclab.cli.load_external_tasks)
    entry_points={"isaaclab.tasks": ["arti_h1 = arti_h1.tasks"]},
    python_requires=">=3.12",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 6.2.0",
    ],
    zip_safe=False,
)
