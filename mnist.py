import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np


MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"


def find_cached_file(filename):
    cache_dir = Path.home() / "tensorflow_datasets" / "downloads" / "mnist"
    if not cache_dir.exists():
        return None

    stem = filename.removesuffix(".gz")
    matches = [
        path
        for path in cache_dir.glob(f"*{stem}*.gz")
        if path.is_file() and not path.name.endswith(".INFO")
    ]
    return matches[0] if matches else None


def get_mnist_file(filename):
    cached = find_cached_file(filename)
    if cached is not None:
        return cached

    raw_dir = Path("mnist_raw")
    raw_dir.mkdir(exist_ok=True)
    path = raw_dir / filename
    if not path.exists():
        url = f"{MNIST_BASE_URL}/{filename}"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def read_images(path):
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file")
        data = np.frombuffer(f.read(), dtype=np.uint8)

    expected = count * rows * cols
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} pixels, expected {expected}")
    return data.reshape(count, rows, cols, 1).astype(np.float32) / 255.0


def read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file")
        data = np.frombuffer(f.read(), dtype=np.uint8)

    if data.size != count:
        raise ValueError(f"{path} has {data.size} labels, expected {count}")
    return data.astype(np.float32)


train_images = read_images(get_mnist_file(MNIST_FILES["train_images"]))
train_labels = read_labels(get_mnist_file(MNIST_FILES["train_labels"]))
test_images = read_images(get_mnist_file(MNIST_FILES["test_images"]))
test_labels = read_labels(get_mnist_file(MNIST_FILES["test_labels"]))

train_images.tofile("train_images.mat")
train_labels.tofile("train_labels.mat")
test_images.tofile("test_images.mat")
test_labels.tofile("test_labels.mat")

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
