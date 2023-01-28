from setuptools import setup, find_packages

setup(
    name="servir-mekong-knn",
    version="0.1.0",
    url="http://github.com/lemma-osu/servir-mekong-knn/",
    author="LEMMA group @ Oregon State University",
    author_email="matt.gregory@oregonstate.edu",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    description="Python based nearest neighbor mapping for SERVIR Mekong",
    install_requires=["earthengine-api", "gee-knn-python"],
    # extras_require={"test": ["pytest", "pytest-cov", "tox"]},
)
