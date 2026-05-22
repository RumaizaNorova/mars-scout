from setuptools import setup

package_name = "mars_scout_geometry"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/mars_scout_geometry"]),
        ("share/mars_scout_geometry", ["package.xml"]),
    ],
        install_requires=["setuptools", "numpy", "opencv-python"],
    entry_points={
        "console_scripts": [
            "projection_node = mars_scout_geometry.projection_node:main",
        ],
    },
)
