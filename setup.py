from setuptools import setup
setup(name='pybarnes',version='0.2.0',author='Lin Ouyang',
    packages=["pybarnes"],
    include_package_data=False,
    install_requires=["metpy", "tqdm", "psutil"]
)
