import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from utils import local_IoU


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

    def local_IoU(self):
        """This function calculates the IoU for the image i.

        Args:
            xmin_pred_i: Value of the prediction min x-axis.
            xmax_pred_i: Value of the prediction max x-axis.
            ymin_pred_i: Value of the prediction min y-axis.
            ymax_pred_i: Value of the prediction max y-axis.
            xmin_true_i: Value of the true min x-axis.
            xmax_true_i: Value of the true max x-axis.
            ymin_true_i: Value of the true min y-axis.
            ymax_true_i: Value of the true max y-axis.

        Returns:
            The return value is the intersection over union.

        """
        if (xmin_true_i, xmax_true_i, ymin_true_i, ymax_true_i) == (0, 0, 0, 0):
            if (xmin_pred_i, xmax_pred_i, ymin_pred_i, ymax_pred_i) == (
                0,
                0,
                0,
                0,
            ):
                return 1

            else:
                return 0

        else:
            box_pred_i = [
                [xmin_pred_i, ymin_pred_i],
                [xmax_pred_i, ymin_pred_i],
                [xmax_pred_i, ymax_pred_i],
                [xmin_pred_i, ymax_pred_i],
            ]
            box_true_i = [
                [xmin_true_i, ymin_true_i],
                [xmax_true_i, ymin_true_i],
                [xmax_true_i, ymax_true_i],
                [xmin_true_i, ymax_true_i],
            ]
            poly_1 = Polygon(box_pred_i)
            poly_2 = Polygon(box_true_i)
            try:
                iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
                return iou
            except:
                return 0


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
