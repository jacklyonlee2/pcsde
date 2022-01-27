# Set Diffusion Network

## Installation
* Set up and activate conda environment.

```shell
conda env create -f environment.yml
conda activate setdiffusion
```

* Compile CUDA extensions.

```shell
sh scripts/install.sh
```

* Download ShapeNet dataset and trained checkpoints.

```shell
sh scripts/download.sh
```

## Training
You can train using `train.py` or provided scripts.

```shell
# Train using CLI
python train.py --name NAME --cate airplane
# Train using provided settings
sh scripts/train_shapenet_aiplane.sh
sh scripts/train_shapenet_car.sh
sh scripts/train_shapenet_chair.sh
```

## Testing
You can evaluate checkpointed models using `test.py` or provided scripts.

```shell
# Test user specified checkpoint using CLI
python test.py --ckpt_path CKPT_PATH --cate car
# Test provided checkpoints
sh scripts/test_shapenet_aiplane.sh
sh scripts/test_shapenet_car.sh
sh scripts/test_shapenet_chair.sh
```
