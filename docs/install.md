# Environment

It is recommanded to build a new virtual environment.

## 1. Install pytorch and requirements.

```bash
# first install pytorch
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```


## 2. Install mm packages

### 2.1 Install `mmcv-full`

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv && git checkout v1.6.0
FORCE_CUDA=1 MMCV_WITH_OPS=1 python -m pip install .
```

### 2.2 Install other mm packages

```bash
python -m pip install mmdet==2.26.0 mmseg==0.28.0 mmdet3d==1.0.0rc6
```

## 3. Install other requirements

```bash
# then clone DV-3DLane and change directory to it to install requirements
cd ${DV-3DLane}
python -m pip install -r requirements.txt

# install pillar ops:
cd modes/ops/pillar_ops && python setup.py build_ext --inplace
```
