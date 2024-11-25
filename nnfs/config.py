import os 
from pathlib import Path



MAIN_DIR = Path(os.path.abspath(__file__)).parent
DATASETS_DIR = MAIN_DIR.joinpath('datasets')

mnist_name = 'fashion_mnist_images'
mnist_url = 'https://nnfs.io/datasets/fashion_mnist_images.zip'