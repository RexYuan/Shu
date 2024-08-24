from setuptools import setup, find_packages

with open("src/README.md", "r") as f:
    long_description = f.read()

setup(
    name="fairness_checker",
    version="0.1.12",
    package_dir={'': 'src'},
    packages=find_packages(where="src"),
    description="Fairnes checker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RexYuan/Shu",
    author="RexYuan",
    author_email="hello@rexyuan.com",
    license="Unlicense",
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'fairness_checker=fairness_checker.__main__:main'
        ]
    },
    package_data={
        'fairness_checker': ['py.typed'],
    },
)
