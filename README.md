# PrototypeDL Hydra lightning adaptation

The hydra lightning implemention of "Deep Learning for Case-based Reasoning through Prototypes: A Neural Network that Explains Its Predictions."

Adapted from [https://github.com/mostafij-rahman/PyTorch-PrototypeDL](https://github.com/mostafij-rahman/PyTorch-PrototypeDL)

Generated from template [https://github.com/ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
For details on file structure, see tempalte page.

This implementation uses WANDB as logger.




## 🚀  Quickstart

```bash
# clone project
git clone https://github.com/ashleve/lightning-hydra-template
cd lightning-hydra-template

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```
Or install manually with conda based of environment.yaml.

### WANDB setup
Requires wandb login, see [quick start](https://docs.wandb.ai/quickstart)


```bash
# train on CPU, by default
python .\src\proto.py

# train on 1 GPU
python .\src\proto.py trainer=proto_gpu
```

