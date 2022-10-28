import os
from os.path import join
import torchvision.datasets as datasets
from loguru import logger

mnist_dataset_train = datasets.MNIST(root="./mnist_data", train=True, download=True)
mnist_dataset_test = datasets.MNIST(root="./mnist_data", train=False, download=True)

logger.info("num train {}".format(len(mnist_dataset_train)))
logger.info("num test {}".format(len(mnist_dataset_test)))

OUTPUT_FOLDER = join("mnist_data", "raw")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
train_folder = join(OUTPUT_FOLDER, "train")
test_folder = join(OUTPUT_FOLDER, "test")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

for i in range(len(mnist_dataset_train)):
    img, tgt = mnist_dataset_train.__getitem__(i)
    output_path = join(train_folder, "image_"+str(i)+"__"+str(tgt)+".jpg")
    img.save(output_path)

for i in range(len(mnist_dataset_test)):
    img, tgt = mnist_dataset_test.__getitem__(i)
    output_path = join(test_folder, "image_"+str(i)+"__"+str(tgt)+".jpg")
    img.save(output_path)