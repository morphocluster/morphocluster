import os
import yaml

MORPHOCLUSTER_CUDA = os.environ["MORPHOCLUSTER_CUDA"]

if MORPHOCLUSTER_CUDA == "no":
    update = {"dependencies": ["cpuonly"]}
else:
    update = {"dependencies": [f"cudatoolkit={MORPHOCLUSTER_CUDA}"]}

with open("environment.update.yml", "w") as f:
    yaml.dump(update, f)

os.system("conda-merge environment.base.yml environment.update.yml > environment.yml")
