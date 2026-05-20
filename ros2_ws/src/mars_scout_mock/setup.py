from setuptools import setup, find_packages

package_name = "mars_scout_mock"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    install_requires=["setuptools", "numpy", "opencv-python"],
    entry_points={
        "console_scripts": [
            "mock_bridge = mars_scout_mock.mock_bridge_node:main",
        ],
    },
)
