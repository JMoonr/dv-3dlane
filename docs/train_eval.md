# Training and Evaluation

## Train


```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node 4 main.py --config config/release_iclr/dv3dlane_openlane1000_base.py
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node 4 main.py --config config/release_iclr/dv3dlane_openlane1000_base.py --cfg-options evaluate=true eval_ckpt={YOUR MODEL PATH}
```