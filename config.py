import os

# Override with `export SMILE_DATA_ROOT=/path/to/datasets` (e.g. on CARC)
data_root = os.environ.get('SMILE_DATA_ROOT', '/data')

# -----------------
# DATASET ROOTS
# -----------------
cifar_10_root = os.path.join(data_root, 'cifar10')
cifar_100_root = os.path.join(data_root, 'cifar100')
cub_root = os.path.join(data_root, 'cub')
aircraft_root = os.path.join(data_root, 'aircraft')
herbarium_dataroot = os.path.join(data_root, 'herb')
imagenet_root = os.path.join(data_root, 'ImageNet')
pets_root = os.path.join(data_root, 'oxford_pets')          # contains oxford-iiit-pet/{images,annotations}

# OSR Split dir
osr_split_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ssb_splits')

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path = os.path.join(data_root, 'dino', 'dino_vitbase16_pretrain.pth')
exp_root = os.environ.get('SMILE_EXP_ROOT', '/home/duruoyi/OCD/')          # All logs and checkpoints will be saved here
