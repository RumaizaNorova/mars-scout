from setuptools import setup

package_name = "rover_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    install_requires=["setuptools"],
    entry_points={
        "console_scripts": [
            "agent_node = rover_nav.agent_node:main",
        ],
    },
)
