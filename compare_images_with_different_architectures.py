import glob
import os
import time

import cv2
import pandas as pd
import matplotlib.pyplot as plt

from image_util import zoom_image, add_noise, mirror_image, blur_image, rotate_image, adjust_brightness, adjust_contrast
from similarity_orb import compare_images_with_orb
from similarity_sift import compare_images_with_sift
from similarity_surf import compare_images_with_surf


# Orjinal ve Dönüştürülmüş Görüntüler Arasındaki Benzerlik Skorlarını Hesapla
def compare_images(img_paths):
    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        transformations = {
            "Original": img,
            "Rotate 45": rotate_image(img, 45),
            "Rotate 90": rotate_image(img, 90),
            "Rotate 135": rotate_image(img, 135),
            "Rotate 180": rotate_image(img, 180),
            # "Mirror": mirror_image(img),
            "Blur": blur_image(img),
            # "Noise": add_noise(img),
            "Zoom In": zoom_image(img, 1.5),
            "Zoom Out": zoom_image(img, 0.5),
            "Brightness Up": adjust_brightness(img, 1.5),
            "Brightness Down": adjust_brightness(img, 0.5),
            "Contrast Up": adjust_contrast(img, 50),
            "Contrast Down": adjust_contrast(img, -50)
        }

        all_results = []
        for name, transformed_img in transformations.items():
            start_time = time.time()
            sift_score = compare_images_with_sift(img, transformed_img)
            sift_duration = time.time() - start_time

            start_time = time.time()
            surf_score = compare_images_with_surf(img, transformed_img)
            surf_duration = time.time() - start_time

            start_time = time.time()
            orb_score = compare_images_with_orb(img, transformed_img)
            orb_duration = time.time() - start_time

            all_results.append((img_path, name, sift_score, surf_score, orb_score, sift_duration, surf_duration, orb_duration))

    data_frame = pd.DataFrame(all_results, columns=["Image",
                                                    "Transformation",
                                                    "SIFT Score",
                                                    "SURF Score",
                                                    "ORB Score",
                                                    "SIFT Duration",
                                                    "SURF Duration",
                                                    "ORB Duration"])

    return data_frame


if __name__ == "__main__":
    img_dir = "dataset_images"
    img_paths = (glob.glob(os.path.join(img_dir, "*.jpg")) +
                 glob.glob(os.path.join(img_dir, "*.png")) +
                 glob.glob(os.path.join(img_dir, "*.jpeg")))

    start_time = time.time()
    print(str(start_time))
    df = compare_images(img_paths)
    print(str(time.time() - start_time))

    df.to_csv("similarity_results_2.csv", index=False)

    # Ortalama değerleri hesapla
    avg_df = df.groupby("Transformation").mean().reset_index()

    colors = ['#26547c', '#ef476f', '#ffd166']

    # Benzerlik Skorları Grafiği
    avg_df.plot(x="Transformation", y=["SIFT Score", "SURF Score", "ORB Score"], kind="bar", figsize=(10, 6), color=colors)
    plt.title("Görüntü Dönüşümleri Arasında Benzerlik Skorları")
    plt.ylabel("Benzerlik Skoru")
    plt.xlabel("Dönüşüm")
    plt.show()

    # Süre Grafiği
    avg_df.plot(x="Transformation", y=["SIFT Duration", "SURF Duration", "ORB Duration"], kind="bar", figsize=(10, 6), color=colors)
    plt.title("Görüntü Dönüşümleri Arasında Algoritma Süreleri")
    plt.ylabel("Süre (saniye)")
    plt.xlabel("Dönüşüm")
    plt.show()
