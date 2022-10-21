#!/usr/bin/env python
# coding: utf-8

# imports
import pandas as pd
import os
import shutil
import urllib.request
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from collections import Counter

#!pip install --upgrade imutils
# import the necessary packages
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# util functions
def clean_label_data(data_path: str):
    data = pd.read_csv(data_path, sep=",", header=0, encoding="utf-8")
    data = data[~data.algae_detail_photo.isnull()]
    data["label"] = data["status"].apply(
        lambda x: 1 if "good_data" in str(x) else (0 if "bad_data" in str(x) else -1)
    )
    data = data[data.label.isin([0, 1])]
    return data


def extract_images(data_df: pd.DataFrame, output_path: str):

    for label in data.label.unique():
        label_path = Path(output_path, str(label))
        if os.path.isdir(label_path):
            shutil.rmtree(label_path)
        os.mkdir(label_path)

    for ix, row in tqdm(data.iterrows(), total=len(data)):
        try:
            # overview_url = row["algae_overview_photo"]
            detailed_url = row["algae_detail_photo"]
            # if isinstance(overview_url, str):
            #    overview_url = urllib.parse.quote(overview_url,safe=':/') # <- here
            if isinstance(detailed_url, str):
                detailed_url = urllib.parse.quote(detailed_url, safe=":/")  # <- here
        except:
            print("Failed name change", overview_url, detailed_url)
        try:
            # if isinstance(overview_url, str):
            #    urllib.request.urlretrieve(overview_url,
            #                               Path(f"../processed_data/images/{int(row['label'])}",
            #                               str(row["label"])+"_"+os.path.basename(overview_url)))
            if isinstance(detailed_url, str):
                urllib.request.urlretrieve(
                    detailed_url,
                    Path(
                        Path(output_path, str(row["label"])),
                        str(row["label"]) + "_" + os.path.basename(detailed_url),
                    ),
                )
        except:
            pass
            # print("Failed to extract image", detailed_url)
            # print(overview_url, detailed_url)


def get_main_color(file):
    img = Image.open(file)
    colors = Counter(img.getdata())  # dict: color -> number
    return max(colors, key=colors.get), len(set(colors))  # most frequent color


def filter_by_color(folder_path: str):
    for image in tqdm(Path(folder_path).glob("**/*")):
        if image.endswith((".jpg", ".png", ".jpeg")) and image.is_file():
            color_tuple, n_cols = get_main_color(image)
            if color_tuple in [
                (0, 0, 0),
                (255, 255, 255),
                (255, 255, 255, 255),
                (0, 0, 0, 0),
            ]:
                print(f"Removing {image}")
                os.remove(image)


def resize_and_crop(folder_path: str, new_size: tuple = (300, 300)):
    for image in tqdm(Path(folder_path).glob("**/*")):
        if image.endswith((".jpg", ".png", ".jpeg", ".JPG")) and image.is_file():
            im = Image.open(image)
            # im = im.resize(new_size)
            width, height = im.size  # Get dimensions
            new_width, new_height = new_size

            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2

            # Crop the center of the image
            im = im.crop((left, top, right, bottom))
            im = im.save(image)
        else:
            os.remove(image)


def image_colorfulness(image):
    # split the image into its respective RGB components
    try:
        (B, G, R) = cv2.split(image.astype("float"))
    except ValueError:
        (B, G, R, A) = cv2.split(image.astype("float"))
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)


def filter_by_colorfulness(folder_path):
    for image in tqdm(Path(folder_path).glob("**/*")):
        if image.endswith((".jpg", ".png", ".jpeg")):
            im = np.array(Image.open(image))
            c = image_colorfulness(im)
            # TODO: Find a more objective measure of colorfulness
            if c > 50:
                print("Removing", image)
                os.remove(image)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", help="provide location of csv from Maranics database", type=str
    )
    parser.add_argument(
        "--save_path", help="provide location to save extracted images", type=str
    )
    parser.add_argument(
        "--crop_size",
        help="provide size of crop to be performed at center of image",
        type=int,
        required=False,
    )

    args = parser.parse_args()

    # TODO: Combine into pipeline where each transform is performed in sequence for all

    cleaned_data = clean_label_data(args.data_path)
    extract_images(cleaned_data, args.save_path)
    resize_and_crop(args.save_path, args.crop_size)
    filter_by_color(args.save_path)
    filter_by_colorfulness(args.save_path)
