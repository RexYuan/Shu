from setuptools import setup, find_packages

with open("src/README.md", "r") as f:
    long_description = f.read()

setup(
    name="fairness_checker",
    version="0.0.2",
    package_dir={'': 'src'},
    packages=find_packages(where="src"),
    description="Fairnes checker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RexYuan/Shu",
    author="RexYuan",
    author_email="hello@rexyuan.com",
    license="Unlicense",
)
