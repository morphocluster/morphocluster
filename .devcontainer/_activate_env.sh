eval "$(micromamba shell hook --shell=bash -p /opt/conda)"

# For robustness, try all possible activate commands.
conda activate base 2>/dev/null \
  || mamba activate base 2>/dev/null \
  || micromamba activate base