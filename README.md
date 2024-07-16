# PrototypeDL Hydra lightning adaptation

The hydra lightning implemention of "Deep Learning for Case-based Reasoning through Prototypes: A Neural Network that Explains Its Predictions."

Adapted from [https://github.com/mostafij-rahman/PyTorch-PrototypeDL](https://github.com/mostafij-rahman/PyTorch-PrototypeDL)

Generated from template [https://github.com/ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
For details on file structure, see template page.

This implementation uses WANDB as logger.




## 🚀  Quickstart

```bash
# clone project
git clone https://github.com/xiguang182/lightning-PrototypeDL.git
cd lightning-PrototypeDL

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

```bash
!pip install wandb
wandb login
```
Requires wandb login for online logging, see [quick start](https://docs.wandb.ai/quickstart)
Or config with other loggers. Refer to template page

### Training
```bash
# train on CPU, by default
python .\src\proto.py

# train on 1 GPU
python .\src\proto.py trainer=proto_gpu
```

