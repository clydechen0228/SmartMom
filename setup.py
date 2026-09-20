from setuptools import setup, find_packages

setup(
    name="laya",
    version="0.3.4",
    description="Fast, non-autoregressive System 1 decision engine with calibrated probabilities",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Convai Innovations",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.45.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.20.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "serve": ["fastapi>=0.110.0", "uvicorn>=0.27.0"],
    },
    entry_points={
        "console_scripts": ["laya-serve = laya.serve:main"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
