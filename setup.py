"""FD-LLM: Multi-Sensor Slurry Fault Diagnosis System"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="fd-llm",
    version="0.1.0",
    description="Hybrid time-series classification and LLM explanation system for slurry pipeline fault diagnosis",
    author="RedMeters",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "isort", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "fd-llm-train=training.train_rocket:main",
            "fd-llm-explain=explainer.run_explainer:main",
            "fd-llm-eval=evaluation.metrics:main",
        ],
    },
)

