#!/bin/bash
#Ensure that the version is updated in pyproject.toml

# 1) Clean any old artifacts
rm -rf dist build *.egg-info

# 2) Rebuild (this will produce files named ...-1.0.0.post1-...
python -m pip install --upgrade build twine
python -m build

# 3) Sanity-check that the filenames reflect the new version
ls dist
# expect: vecclean-1.0.0.post1-...whl and vecclean-1.0.0.post1.tar.gz

# (optional) Check long-description renders
twine check dist/*

# 4) Upload to TestPyPI
twine upload -r testpypi dist/* --verbose
