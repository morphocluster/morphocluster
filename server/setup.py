from setuptools import setup

import versioneer

setup(
    name="morphocluster",
    packages=["morphocluster"],
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    include_package_data=True,
    entry_points={"console_scripts": ["morphocluster = morphocluster.scripts:main"]},
)
