import os 
from pathlib import Path



MAIN_DIR = Path(os.path.abspath(__file__)).parent
DATASETS_DIR = MAIN_DIR.joinpath('datasets')

mnist_name = 'fashion_mnist_images'
mnist_url = 'https://nnfs.io/datasets/fashion_mnist_images.zip'

mnist_labels = {
    0 : 'T-shirt/top',
    1 : 'Trouser',
    2 : 'Pullover',
    3 : 'Dress',
    4 : 'Coat',
    5 : 'Sandal',
    6 : 'Shirt',
    7 : 'Sneaker',
    8 : 'Bag',
    9 : 'Ankle boot',
    }