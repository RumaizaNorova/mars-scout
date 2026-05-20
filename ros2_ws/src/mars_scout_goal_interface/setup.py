from setuptools import setup, find_packages
setup(
    name="mars_scout_goal_interface",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    install_requires=["setuptools"],
    entry_points={"console_scripts": [
        "goal_cli = mars_scout_goal_interface.goal_cli:main",
    ]},
)
