"""
AIoT 时空预测模型项目安装配置

Author: AI Assistant
Date: 2025-07-29
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# 读取requirements文件
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="aiot-spatiotemporal-parking-forecaster",
    version="1.0.0",
    author="AI Assistant",
    author_email="ai@example.com",
    description="AIoT时空预测模型本地复现项目 - 停车场占用率预测",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/aiot-spatiotemporal-parking-forecaster",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
            "flake8>=6.0.0",
            "pre-commit>=3.3.0",
        ],
        "viz": [
            "plotly>=5.15.0",
            "dash>=2.10.0",
        ],
        "experiment": [
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "aiot-train=scripts.train:main",
            "aiot-inference=scripts.inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
    zip_safe=False,
    keywords=[
        "deep learning",
        "graph neural network",
        "transformer",
        "time series",
        "parking prediction",
        "spatiotemporal",
        "pytorch"
    ],
    project_urls={
        "Bug Reports": "https://github.com/example/aiot-spatiotemporal-parking-forecaster/issues",
        "Source": "https://github.com/example/aiot-spatiotemporal-parking-forecaster",
        "Documentation": "https://github.com/example/aiot-spatiotemporal-parking-forecaster/docs",
    },
)
