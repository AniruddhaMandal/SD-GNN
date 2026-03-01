# SS-GNN/src/gxl/setup.py
from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent.resolve()

# Prefer monorepo root README; fallback to local README; else empty.
candidates = [here / "../../README.md", here / "README.md"]
long_description = ""
long_description_content_type = "text/plain"
for p in candidates:
    try:
        long_description = Path(p).read_text(encoding="utf-8")
        long_description_content_type = "text/markdown"
        break
    except Exception:
        pass

# If you keep a single license at repo root, copy/symlink it to src/gxl/LICENSE so it’s included.
license_file = here / "LICENSE"
license_files = ["LICENSE"] if license_file.exists() else []

setup(
    name="gxl",                      
    version="1.5.0",
    description="Graph neural network training framework with subgraph sampling",
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    author="Aniruddha Mandal",
    author_email="ani96dh@gmail.com",
    url="https://github.com/AniruddhaMandal/SS-GNN",
    project_urls={
        "Source": "https://github.com/AniruddhaMandal/SS-GNN",
        "Issues": "https://github.com/AniruddhaMandal/SS-GNN/issues",
    },
    license="MIT",
    license_files=license_files,
    keywords=["graph neural networks", "GNN", "subgraph sampling", "PyTorch", "PyG"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    packages=find_packages(exclude=("tests", "docs", "examples")),
    include_package_data=True,
    install_requires=[
        "numpy>=1.26",
        "pyyaml>=6.0",
        "tqdm>=4.62",
    ],
    extras_require={
        "gnn": [
            "torch>=2.2",
            "torch-geometric>=2.4.0",
        ],
        "viz": ["matplotlib>=3.7"],
        "sampler": [
            # Add PyPI name here if/when ugs_sampler is published, e.g.:
            # "ugs-sampler>=0.1.0"
        ],
        "dev": ["black>=24.0", "ruff>=0.4", "mypy>=1.8", "pre-commit>=3.5"],
        "test": ["pytest>=7.4", "pytest-cov>=4.1"],
        "docs": ["mkdocs>=1.5", "mkdocs-material>=9.5"],
        "all": [
            "torch>=2.2",
            "torch-geometric>=2.4.0",
            "matplotlib>=3.7",
            "black>=24.0",
            "ruff>=0.4",
            "mypy>=1.8",
            "pre-commit>=3.5",
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "mkdocs>=1.5",
            "mkdocs-material>=9.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "gxl=gxl.cli:main",
        ]
    },
    package_data={
        "gxl": ["py.typed"],
    },
    zip_safe=False,
)
