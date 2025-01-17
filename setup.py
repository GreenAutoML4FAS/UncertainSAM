from setuptools import setup, find_packages

setup(
    name="CertainSAM",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[],
    author="...",
    author_email="...",
    description="CertainSAM",
    long_description="CertainSAM: Fast and Efficient Uncertainty Quantification in the Segment Anything Model",
    entry_points={
        'console_scripts': [
            'certainsam_demo = usam.scripts.demo:main',
        ],
    },
)
