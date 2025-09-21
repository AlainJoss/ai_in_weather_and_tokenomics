instructions: https://nvidia.github.io/earth2studio/userguide/developer/overview.html
```bash
# clone: add " e2s" to the statement if you already have a ai-in-weather-and-tokenomics folder
git clone git@github.com:AlainJoss/ai-in-weather-and-tokenomics.git

# create environment: takes forever (15min)
uv venv --python=3.12
source .venv/bin/activate
uv sync --extra fcn --extra fcn3 --extra sfno

# run main once everything is installed
# loading packages takes forever, but once the thing runs, it's fast!
cd src
uv run main.py
```

run fcn3 --> this doesn't work but I don't know how to make it work:
```bash
export FORCE_CUDA_EXTENSION=1
uv pip install --no-build-isolation torch-harmonics
# maybe this works, the first command continues to raise a warning
pip install --no-build-isolation torch-harmonics
```