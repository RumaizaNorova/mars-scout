from setuptools import setup, find_packages
setup(
    name="mars_scout_control",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    install_requires=["setuptools", "numpy"],
    entry_points={"console_scripts": [
        "agent_node = mars_scout_control.agent_node:main",
    ]},
)
