import cv2
import numpy as np


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def mirror_image(image):
    return cv2.flip(image, 1)


def blur_image(image, ksize=5):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def add_noise(image, mean=0, std=25):
    gauss = np.random.normal(mean, std, image.shape).astype('uint8')
    noisy = cv2.add(image, gauss)
    return noisy


def zoom_image(image, zoom_factor):
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    radius_x, radius_y = int(center_x / zoom_factor), int(center_y / zoom_factor)
    min_x, max_x = center_x - radius_x, center_x + radius_x
    min_y, max_y = center_y - radius_y, center_y + radius_y
    cropped = image[min_y:max_y, min_x:max_x]
    return cv2.resize(cropped, (w, h))


def adjust_brightness(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def adjust_contrast(image, factor):
    return cv2.convertScaleAbs(image, alpha=1, beta=factor)
