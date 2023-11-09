import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from utils import local_IoU, unnormalize_boxes
import torch


class ResultsEvaluation:
    def __init__(self, results):
        self.results = results  # self.results from Yolov8 model
        self.filenames = [result.path.split("/")[-1] for result in self.results]

    def get_results(self, annotation_completed):
        tab_results = np.empty((len(self.filenames), 10))
        annotation = annotation_completed[
            annotation_completed["im_name"].isin(self.filenames)
        ].copy()
        for i, filename in enumerate(self.filenames):
            tab_results[i, 0] = annotation[annotation["im_name"] == filename][
                "class"
            ].values[0]
            tab_results[i, 1:5] = annotation.values[i, 2:6]
            if len(self.results[i].boxes.cls) != 0:
                tab_results[i, 5] = 1
                tab_results[i, 6:] = self.results[i].boxes.xyxy[0]
            else:
                tab_results[i, 5] = 0
                tab_results[i, 6:] = [0, 0, 0, 0]
        self.df_results = pd.DataFrame(
            tab_results,
            columns=[
                "class",
                "x1",
                "y1",
                "x2",
                "y2",
                "pred",
                "pred_x1",
                "pred_y1",
                "pred_x2",
                "pred_y2",
            ],
        )
        self.df_results["im_name"] = annotation["im_name"].values
        self.df_results["IoU"] = self.df_results.apply(
            lambda row: local_IoU(
                row["x1"],
                row["x2"],
                row["y1"],
                row["y2"],
                row["pred_x1"],
                row["pred_x2"],
                row["pred_y1"],
                row["pred_y2"],
            ),
            axis=1,
        )
        return self.df_results

    def get_results_unormalized(
        self, annotation_completed, original_size, current_size
    ):
        if self.df_results is None:
            self.get_results(annotation_completed)
        self.df_results[["x1", "x2", "y1", "y2"]] = pd.DataFrame(
            self.df_results.apply(
                lambda row: np.array(
                    unnormalize_boxes(
                        row["x1"],
                        row["x2"],
                        row["y1"],
                        row["y2"],
                        current_size,
                        original_size.loc[row["im_name"]],
                    )
                ),
                axis=1,
            ).to_list(),
            columns=["x1", "x2", "y1", "y2"],
        )

        self.df_results[["pred_x1", "pred_x2", "pred_y1", "pred_y2"]] = pd.DataFrame(
            self.df_results.apply(
                lambda row: np.array(
                    unnormalize_boxes(
                        row["pred_x1"],
                        row["pred_x2"],
                        row["pred_y1"],
                        row["pred_y2"],
                        current_size,
                        original_size.loc[row["im_name"]],
                    )
                ),
                axis=1,
            ).to_list(),
            columns=["pred_x1", "pred_x2", "pred_y1", "pred_y2"],
        )

        return self.df_results


class ResultsVisualisation:
    def __init__(self, results):
        self.results = results  # self.results from Yolov8 model

    def visualize_predictions(self, index_images):
        class_names = self.results[0].names
        for i in index_images:
            # Construct the full path to the image file
            image_path = self.results[i].path
            # Load the image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get the image shape (height, width)
            image_shape = image.shape[:2]

            # Create an array of colors for bounding boxes
            num_classes = len(class_names)
            colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
            result = self.results[i]
            # Iterate over the predictions and draw bounding boxes

            if len(result.boxes.cls) == 0:
                continue
            class_id = int(result.boxes.cls[0].int())
            confidence = float(result.boxes[0].conf)
            x1 = int(result.boxes.xyxy[0][0])
            y1 = int(result.boxes.xyxy[0][1])
            x2 = int(result.boxes.xyxy[0][2])
            y2 = int(result.boxes.xyxy[0][3])

            # # Scale the bounding box coordinates to match the image shape
            # x = int(x * image_shape[1])
            # y = int(y * image_shape[0])
            # width = int(width * image_shape[1])
            # height = int(height * image_shape[0])

            # Get the class name
            class_name = class_names[class_id]

            # Get the color for the bounding box
            color = colors[class_id]

            # Draw the bounding box rectangle and label on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), tuple(color.tolist()), 2)
            label = f"{class_name}: {confidence:.2f}"

            # Display the image
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            plt.axis("off")
            plt.show()


class EvalWithoutScaling:
    def __init__(self, results, path_annotation, filename_annotation):
        self.results = results
        self.path_annotation = path_annotation
        self.filename_annotation = filename_annotation

    def get_results_as_dataframe(self):
        filenames = [result.path.split("/")[-1] for result in self.results]
        annotation = pd.read_csv(self.path_annotation + self.filename_annotation)
        self.df_results = annotation[annotation["im_name"].isin(filenames)].reset_index(
            drop=True
        )
        self.df_results[
            ["pred_class", "pred_x1", "pred_y1", "pred_x2", "pred_y2"]
        ] = pd.DataFrame(
            np.array(
                [
                    np.array(
                        torch.cat(
                            (torch.Tensor([result.boxes.cls[0]]), result.boxes.xyxy[0])
                        )
                    )
                    if len(result.boxes.cls) != 0
                    else [0, 0, 0, 0, 0]
                    for result in self.results
                ]
            ),
            columns=["pred_class", "pred_x1", "pred_y1", "pred_x2", "pred_y2"],
        )
        self.df_results.dropna(subset=["x_min"], inplace=True)
        return self.df_results

    def compute_IoU(self):
        self.df_results["IoU"] = self.df_results.apply(
            lambda row: local_IoU(
                row["x_min"],
                row["x_max"],
                row["y_min"],
                row["y_max"],
                row["pred_x1"],
                row["pred_x2"],
                row["pred_y1"],
                row["pred_y2"],
            ),
            axis=1,
        )
        return self.df_results
