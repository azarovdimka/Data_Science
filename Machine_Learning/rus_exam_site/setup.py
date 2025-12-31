# TODO проверить используется ли файл

from setuptools import setup, find_packages

setup(
    name="rus_exam_site",
    version="1.0.0",
    python_requires=">=3.10.12",
    packages=find_packages(),
    install_requires=[
        line.strip() 
        for line in open("requirements.txt").readlines() 
        if line.strip() and not line.startswith("#")
    ],
)