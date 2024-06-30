import versioneer
from setuptools import find_namespace_packages, setup

setup(
    name="morphocluster.lib",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_namespace_packages("src/", include=["morphocluster.lib"]),
    package_dir={"": "src"},
    install_requires=["chardet", "click", "numpy", "scikit-learn", "h5py", "hdbscan"],
    entry_points={
        "console_scripts": ["morphocluster = morphocluster.lib.scripts:main"]
    },
)
