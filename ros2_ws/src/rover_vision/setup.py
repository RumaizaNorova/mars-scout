from setuptools import setup

package_name = "rover_vision"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    install_requires=["setuptools"],
    entry_points={
        "console_scripts": [
            "vlm_node = rover_vision.vlm_node:main",
        ],
    },
)
