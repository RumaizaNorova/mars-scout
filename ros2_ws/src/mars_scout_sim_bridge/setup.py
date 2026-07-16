from setuptools import setup, find_packages
import os
from glob import glob

package_name = "mars_scout_sim_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
         glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mars-scout",
    maintainer_email="you@example.com",
    description="Isaac Sim <-> Mars Scout ROS2 bridge",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sim_bridge_node = mars_scout_sim_bridge.sim_bridge_node:main",
            "topic_inspector  = mars_scout_sim_bridge.topic_inspector:main",
        ],
    },
)
