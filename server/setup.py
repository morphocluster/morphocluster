from setuptools import find_packages, setup

import versioneer

setup(
    name="morphocluster.server",
    packages=find_packages("src/", include=["morphocluster.server"]),
    package_dir={"": "src"},
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    include_package_data=True,
    install_requires=["morphocluster.lib"],
)
