from setuptools import setup, find_packages

setup(
    name="caac_project",
    version="0.1.0",
    description="Contextual Abductive Attention for Classification - One-vs-Rest Classifier with Shared Latent Cauchy Vectors",
    author="龚鹤扬 (Heyang Gong)",
    author_email="zj3712@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "flake8>=3.9",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
) 