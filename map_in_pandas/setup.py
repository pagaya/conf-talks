from setuptools import find_packages
from setuptools import setup

setup(
    name="pagaya-mapinpandas",
    version="0.5",
    packages=find_packages(),
    url="https://github.com/pagaya/conf-talks/tree/master/map_in_pandas",
    author="pagaya-tal-franji",
    author_email="tal.franji@pagaya.com",
    description="Easy python wrapper for Spark mapInPandas, applyInPandas",
    package_data={"doc": ["*.ipynb"]},
    keywords='spark pyspark',
)
