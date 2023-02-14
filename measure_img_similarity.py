import os
import random
import shutil
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.metrics import structural_similarity
from skimage.transform import resize
from scipy.stats import wasserstein_distance
import numpy as np
import cv2
from tensorflow.python.data import Dataset
from tqdm import tqdm
from PIL import Image

##
# Globals
##

# specify resized image sizes
height = 2 ** 10
width = 2 ** 10


##
# Functions
##

def get_img(image_array, norm_size=True, norm_exposure=False):
    '''
    Prepare an image for image processing tasks
    '''
    # flatten returns a 2d grayscale array
    # img = imread(path, flatten=True).astype(int)
    img = image_array.flatten()
    # resizing returns float vals 0:255; convert to ints for downstream tasks
    if norm_size:
        img = resize(img, (height, width), anti_aliasing=True, preserve_range=True)
    if norm_exposure:
        img = normalize_exposure(img)
    return img


def get_histogram(img):
    '''
    Get the histogram of an image. For an 8-bit, grayscale image, the
    histogram will be a 256 unit vector in which the nth value indicates
    the percent of the pixels in the image with the given darkness level.
    The histogram's values sum to 1.
    '''
    h, w = img.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
    return np.array(hist) / (h * w)


def normalize_exposure(img):
    '''
    Normalize the exposure of an image.
    '''
    img = img.astype(int)
    hist = get_histogram(img)
    # get the sum of vals accumulated by each position in hist
    cdf = np.array([sum(hist[:i + 1]) for i in range(len(hist))])
    # determine the normalization values for each unit of the cdf
    sk = np.uint8(255 * cdf)
    # normalize each position in the output image
    height, width = img.shape
    normalized = np.zeros_like(img)
    for i in range(0, height):
        for j in range(0, width):
            normalized[i, j] = sk[img[i, j]]
    return normalized.astype(int)


def earth_movers_distance(path_a, path_b):
    '''
    Measure the Earth Mover's distance between two images
    @args:
      {str} path_a: the path to an image file
      {str} path_b: the path to an image file
    @returns:
      TODO
    '''
    img_a = get_img(path_a, norm_exposure=True)
    img_b = get_img(path_b, norm_exposure=True)
    hist_a = get_histogram(img_a)
    hist_b = get_histogram(img_b)
    return wasserstein_distance(hist_a, hist_b)


def structural_distance(image1: np.array, image2: np.array) -> float:
    similarity_index, *_ = structural_similarity(image1.flatten(), image2.flatten(), full=True)
    return -similarity_index


def pixel_sim(path_a, path_b):
    '''
    Measure the pixel-level similarity between two images
    @args:
      {str} path_a: the path to an image file
      {str} path_b: the path to an image file
    @returns:
      {float} a float {-1:1} that measures structural similarity
        between the input images
    '''
    img_a = get_img(path_a, norm_exposure=True)
    img_b = get_img(path_b, norm_exposure=True)
    return np.sum(np.absolute(img_a - img_b)) / (height * width) / 255


def sift_sim(path_a, path_b):
    '''
    Use SIFT features to measure image similarity
    @args:
      {str} path_a: the path to an image file
      {str} path_b: the path to an image file
    @returns:
      TODO
    '''
    # initialize the sift feature detector
    orb = cv2.ORB_create()

    # get the images
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)

    # find the keypoints and descriptors with SIFT
    kp_a, desc_a = orb.detectAndCompute(img_a, None)
    kp_b, desc_b = orb.detectAndCompute(img_b, None)

    # initialize the bruteforce matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # match.distance is a float between {0:100} - lower means more similar
    matches = bf.match(desc_a, desc_b)
    similar_regions = [i for i in matches if i.distance < 70]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)


def throw_images_to_temp_folder(images, temp_folder_path: str, unnormalize: bool):
    temp_folder_path_obj = Path(temp_folder_path)
    if temp_folder_path_obj.exists():
        shutil.rmtree(temp_folder_path_obj)
    temp_folder_path_obj.mkdir(parents=True, exist_ok=True)

    for i, image_tensor in enumerate(images):
        image_array = image_tensor.numpy()[0]
        if unnormalize:
            image_array = (image_array * 127.5 + 127.5).astype(np.uint8)
        Image.fromarray(image_array).save(temp_folder_path_obj / f'{i + 1}.jpg')


# **********************************************************************
def find_competition_dataset_files(local_dataset_folder_path: Path):
    dataset_folder_path = local_dataset_folder_path

    monet_dataset_files = tf.io.gfile.glob(str(dataset_folder_path / 'monet_tfrec/*.tfrec'))
    photo_dataset_files = tf.io.gfile.glob(str(dataset_folder_path / 'photo_tfrec/*.tfrec'))
    assert any(monet_dataset_files)
    assert any(photo_dataset_files)
    print(f"found {len(monet_dataset_files)} monet and {len(photo_dataset_files)} photo tfrec files.")

    return monet_dataset_files, photo_dataset_files


def load_tf_records_dataset(tf_record_files) -> Dataset:
    def _read_and_normalize_tfrecord(record):
        tfrecord_format = {
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.string)
        }
        record = tf.io.parse_single_example(record, tfrecord_format)
        image = record['image']
        image = tf.image.decode_jpeg(image, channels=3)
        # print(f'asdsadsad: {(image.shape, type(image))}')
        # image = tf.keras.utils.array_to_img(image)
        # image = image.resize((320, 320))
        # image = np.array(image)
        # image = tf.convert_to_tensor(image)
        image = (tf.cast(image, tf.float32) / 127.5) - 1
        image = tf.reshape(image, [256, 256, 3])
        image = tf.image.resize(image, (320, 320), method='bilinear')
        return image

    sorted_tf_record_files = sorted(tf_record_files)
    dataset = tf.data.TFRecordDataset(sorted_tf_record_files)
    dataset = dataset.map(_read_and_normalize_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def _incremental_farthest_search(image_tensors_list, k: int, distance_func, resize_before_comparison_shape=None):
    remaining_images = [
        dict(
            img_tensor=img_tensor,
            img_comparison_array=
                tf.image.resize(img_tensor, resize_before_comparison_shape).numpy()[0]
                if resize_before_comparison_shape else img_tensor.numpy()[0]
        )
        for img_tensor in image_tensors_list
    ]

    selected_images = [remaining_images.pop(random.randint(0, len(remaining_images) - 1))]
    for _ in tqdm(list(range(k - 1)), desc='incremental_farthest_search() main loop'):
        distances = [
            distance_func(
                i['img_comparison_array'],
                selected_images[0]['img_comparison_array']
            )
            for i in remaining_images
        ]
        for i, p in enumerate(remaining_images):
            for j, s in enumerate(selected_images):
                distances[i] = min(distances[i], distance_func(
                    p['img_comparison_array'], s['img_comparison_array']
                ))
        selected_images.append(remaining_images.pop(distances.index(max(distances))))
    selected_images = [i['img_tensor'] for i in selected_images]
    return selected_images


def set_training_random_seed(seed: int):
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'set_training_random_seed() - seed value: {seed}')


if __name__ == '__main__':
    set_training_random_seed(42)

    LOCAL_DATASET_FOLDER_PATH = Path('./train_data/gan-getting-started/')
    monet_dataset_files, photo_dataset_files = find_competition_dataset_files(LOCAL_DATASET_FOLDER_PATH)
    original_monet_dataset = load_tf_records_dataset(monet_dataset_files).batch(1)
    # compressed_images = [
    #     tf.image.resize(image_tensor, (100, 100), method='bilinear').numpy()[0]
    #     for image_tensor in original_monet_dataset
    # ]

    farthest_images_list = _incremental_farthest_search(
        list(original_monet_dataset),
        k=30,
        # distance_func=structural_distance,
        distance_func=structural_distance,
        resize_before_comparison_shape=(100, 100)
    )

    throw_images_to_temp_folder(
        farthest_images_list, './tmppppp_images/', unnormalize=True
    )

    # images_shape = list(compressed_images)[0].shape
    # print(f'*** Selected 30 train monet photos (shape: {images_shape}) ***')
    # _, ax = plt.subplots(30, 1, figsize=(50, 50))
    # for i, img in enumerate(compressed_images[:30]):
    #     img = (img * 127.5 + 127.5).numpy()[0].astype(np.uint8)
    #
    #     ax[i].imshow(img)
    # plt.show()

    # img_a = 'a.jpg'
    # img_b = 'b.jpg'
    # # get the similarity values
    # structural_sim = structural_sim(img_a, img_b)
    # pixel_sim = pixel_sim(img_a, img_b)
    # sift_sim = sift_sim(img_a, img_b)
    # emd = earth_movers_distance(img_a, img_b)
    # print(structural_sim, pixel_sim, sift_sim, emd)
