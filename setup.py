from setuptools import setup, find_packages

setup(
    name="phctqa",
    version="1.0.0",
    description="PHCTQA: Physics-informed Head and Thorax CT Quality Assessment",
    author="Your Name",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests", "scripts", "examples"]),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "albumentations>=1.3.0",
        "SimpleITK>=2.3.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
        "openpyxl>=3.1.0",
        "xlwt>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "phctqa-inference=inference:main",
        ],
    },
)