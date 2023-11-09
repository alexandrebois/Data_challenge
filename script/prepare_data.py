import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import shutil
from utils import *


class DataPreparationTrain:
    def __init__(
        self,
        images_path,
        annotation_path,
        annotation_filename,
        im_size=(640, 640),
    ):
        self.annotation_path = annotation_path
        self.images_path = images_path
        self.annotation_filename = annotation_filename
        self.im_size = im_size

    def get_image_original_size(self):
        self.im_original_size = {}
        for file in os.listdir(self.images_path):
            if file.endswith(".jpg"):
                im = plt.imread(self.images_path + file)
                self.im_original_size[file] = im.shape[:2]
        self.im_original_size = pd.DataFrame.from_dict(
            self.im_original_size, orient="index"
        )

    def import_annotation(self):
        annotation = pd.read_csv(self.annotation_path + self.annotation_filename)
        annotation_clean = annotation.dropna(subset=["x_min"])
        self.annotation_clean = annotation_clean

    def scaling(self):
        self.import_annotation()
        self.images_scaled, self.annotation_scaled = normalize_images_and_boxes(
            self.annotation_clean, self.im_size, self.images_path
        )

    def preparing_data_for_YOLO(self):
        if self.annotation_scaled is None:
            self.scaling()
        self.annotation_completed = pd.concat(
            [
                pd.DataFrame(
                    self.annotation_scaled,
                    columns=["x_min", "y_min", "x_max", "y_max"],
                    index=self.annotation_clean.index,
                ),
                self.annotation_clean[["im_name", "class", "models"]],
            ],
            axis=1,
        )[["im_name", "class", "x_min", "y_min", "x_max", "y_max", "models"]]
        self.annotation_completed["x_mid"] = (
            self.annotation_completed["x_min"] + self.annotation_completed["x_max"]
        ) / 2
        self.annotation_completed["y_mid"] = (
            self.annotation_completed["y_min"] + self.annotation_completed["y_max"]
        ) / 2
        self.annotation_completed["w"] = (
            self.annotation_completed["x_max"] - self.annotation_completed["x_min"]
        )
        self.annotation_completed["h"] = (
            self.annotation_completed["y_max"] - self.annotation_completed["y_min"]
        )

        self.annotation_completed.index = range(len(self.annotation_completed))
        self.annotation_completed[["x_mid", "w"]] = (
            self.annotation_completed[["x_mid", "w"]] / self.im_size[0]
        )
        self.annotation_completed[["y_mid", "h"]] = (
            self.annotation_completed[["y_mid", "h"]] / self.im_size[1]
        )

    def download_data_for_training(self, path, val_proportion=0.2):
        if self.annotation_completed is None:
            self.preparing_data_for_YOLO()

        filenames_val = np.random.choice(
            self.annotation_completed["im_name"],
            int(0.2 * len(self.annotation_completed)),
        )

        os.makedirs(path + "/train/images", exist_ok=True)
        os.makedirs(path + "/train/labels", exist_ok=True)
        os.makedirs(path + "/val/images", exist_ok=True)
        os.makedirs(path + "/val/labels", exist_ok=True)

        shutil.rmtree(path + "/train/images")
        shutil.rmtree(path + "/train/labels")
        shutil.rmtree(path + "/val/images")
        shutil.rmtree(path + "/val/labels")

        os.makedirs(path + "/train/images", exist_ok=True)
        os.makedirs(path + "/train/labels", exist_ok=True)
        os.makedirs(path + "/val/images", exist_ok=True)
        os.makedirs(path + "/val/labels", exist_ok=True)

        for i, filename in enumerate(self.annotation_completed["im_name"]):
            if filename in filenames_val:
                plt.imsave(path + "/val/images/" + filename, self.images_scaled[i])
                self.annotation_completed.iloc[i, [1, 7, 8, 9, 10]].to_csv(
                    path + "/val/labels/" + filename[:-4] + ".txt",
                    header=False,
                    index=False,
                    lineterminator=" ",
                )
            else:
                plt.imsave(path + "/train/images/" + filename, self.images_scaled[i])
                self.annotation_completed.iloc[i, [1, 7, 8, 9, 10]].to_csv(
                    path + "/train/labels/" + filename[:-4] + ".txt",
                    header=False,
                    index=False,
                    lineterminator=" ",
                )


# class DataPreparationTest:
#     def __init__(self, images_path):
#         self.images_path = images_path

#     def get_image_original_size(self):
#         self.im_original_size = {}
#         for file in os.listdir(self.images_path):
#             if file.endswith(".jpg"):
#                 im = plt.imread(self.images_path + file)
#                 self.im_original_size[file] = im.shape[:2]
#         self.im_original_size = pd.DataFrame.from_dict(
#             self.im_original_size, orient="index"
#         )
