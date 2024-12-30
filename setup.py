from setuptools import setup, find_packages

print(find_packages())

setup(
    name="prostNfound",
    packages=find_packages(),
    requires=["numpy", "skimage", "simple-parsing"],
)
