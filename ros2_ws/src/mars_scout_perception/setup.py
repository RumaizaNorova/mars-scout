from setuptools import setup, find_packages
setup(
    name="mars_scout_perception",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/mars_scout_perception"]),
        ("share/mars_scout_perception", ["package.xml"]),
    ],
        install_requires=["setuptools", "numpy", "opencv-python"],
    entry_points={"console_scripts": [
        "perception_node = mars_scout_perception.perception_node:main",
    ]},
)
