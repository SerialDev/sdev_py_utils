from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16
import numpy as np
import argparse
import cv2


def convert_cr2_to_jpg(raw_image):
    from rawkit import raw

    raw_image_process = raw.Raw(raw_image)
    buffered_image = numpy.array(raw_image_process.to_buffer())
    if raw_image_process.metadata.orientation == 0:
        jpg_image_height = raw_image_process.metadata.height
        jpg_image_width = raw_image_process.metadata.width
    else:
        jpg_image_height = raw_image_process.metadata.width
        jpg_image_width = raw_image_process.metadata.height
    jpg_image = Image.frombytes('RGB', (jpg_image_width, jpg_image_height), buffered_image)
    return jpg_image


def resize_image(numpy_array_image, new_height):
    # convert nympy array image to PIL.Image
    image = Image.fromarray(numpy.uint8(numpy_array_image))
    old_width = float(image.size[0])
    old_height = float(image.size[1])
    ratio = float(new_height / old_height)
    new_width = int(old_width * ratio)
    image = image.resize((new_width, new_height), PIL.Image.ANTIALIAS)
    # convert PIL.Image into nympy array back again
    return array(image)


def load_bytesio_bytes(data):
    file_bytes = np.asarray(bytearray(data.read()), dtype=np.uint8)
    return file_bytes


def load_bytes_image(file_bytes):
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def load_bytesio_image(data):
    return load_bytes_image(load_bytesio_bytes(data))


def load_bytesio_kerasimg(data):
    image = image_utils.load_img(data, target_size=(224, 224))
    image = image_utils.img_to_array(image)
    return image


def expand_npimg_tfimg(data):
    image = np.expand_dims(data, axis=0)
    return image


def substract_mean_rgbi(data):
    image = preprocess_input(data)
    return image


def initialize_bytesio(obj):
    obj.seek(0)
    return obj


def save_bytesio_file(data, filename="test.png"):
    with open(filename, 'wb') as f:
        f.write(data.read())
    return []


def blur_image(img):
    # Blur image with random kernel
    kernel_size = random.randint(1, 5)
    if kernel_size % 2 != 1:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
