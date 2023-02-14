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


def _calc_image_greyscale_histogram(image: np.array) -> np.array:
    greyscale_image = np.array(Image.fromarray(image).convert('L'))
    h, w = greyscale_image.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[greyscale_image[i, j]] += 1
    return np.array(hist) / (h * w)


def normalize_exposure(img):
    '''
    Normalize the exposure of an image.
    '''
    img = img.astype(int)
    hist = _calc_image_greyscale_histogram(img)
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


def _earth_movers_distance(image1: np.array, image2: np.array) -> float:
    image1_hist = _calc_image_greyscale_histogram(image1)
    image2_hist = _calc_image_greyscale_histogram(image2)
    distance = wasserstein_distance(image1_hist, image2_hist)
    return distance


def _structural_distance(image1: np.array, image2: np.array) -> float:
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


def _incremental_farthest_search(ordered_image_tensors_list, k: int, distance_func, pre_comparison_transformation_func):
    remaining_images = [
        dict(
            orig_image_index=orig_image_index,
            img_tensor=img_tensor,
            img_comparison_array=pre_comparison_transformation_func(img_tensor)
        )
        for orig_image_index, img_tensor in enumerate(ordered_image_tensors_list)
    ]

    chosen_30_images_indices = [remaining_images.pop(random.randint(0, len(remaining_images) - 1))]
    for _ in tqdm(list(range(k - 1)), desc='incremental_farthest_search() main loop'):
        distances = [
            distance_func(
                i['img_comparison_array'],
                chosen_30_images_indices[0]['img_comparison_array']
            )
            for i in remaining_images
        ]
        for i, p in enumerate(remaining_images):
            for j, s in enumerate(chosen_30_images_indices):
                distances[i] = min(distances[i], distance_func(
                    p['img_comparison_array'], s['img_comparison_array']
                ))
        chosen_30_images_indices.append(remaining_images.pop(distances.index(max(distances))))
    chosen_30_images_indices = [i['orig_image_index'] for i in chosen_30_images_indices]
    return chosen_30_images_indices


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

    comparison_images_resize_shape = (100, 100)

    def _pre_comparison_transformation_func(image_tensor):
        image_array = image_tensor.numpy()
        denormalized_image = (image_array * 127.5 + 127.5)
        resized_image = tf.image.resize(denormalized_image, comparison_images_resize_shape).numpy()
        resized_image = resized_image.astype(np.uint8)
        resized_image = resized_image[0]
        return resized_image


    original_ordered_monet_images = list(original_monet_dataset)
    chosen_30_images_indices = _incremental_farthest_search(
        original_ordered_monet_images,
        k=5,
        # distance_func=_structural_distance,
        distance_func=_earth_movers_distance,
        pre_comparison_transformation_func=_pre_comparison_transformation_func
    )
    farthest_images_list = Dataset.from_tensor_slices([
        original_ordered_monet_images[image_idx]
        for image_idx in chosen_30_images_indices
    ])

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
