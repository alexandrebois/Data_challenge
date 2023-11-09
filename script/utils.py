import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from shapely.geometry import Polygon


def normalize_images_and_boxes(bbox_data, target_size, image_directory):
    """
    Normalize the size of images and bounding boxes.

    Args:
    bbox_data (pandas.DataFrame): A DataFrame containing information about images and bounding boxes. It should have the following columns:
        - image_name: Name of the image file.
        - x1: X-coordinate of the top-left corner of the bounding box.
        - y1: Y-coordinate of the top-left corner of the bounding box.
        - x2: X-coordinate of the bottom-right corner of the bounding box.
        - y2: Y-coordinate of the bottom-right corner of the bounding box.
    target_size (tuple): A tuple (width, height) specifying the desired size to which images should be resized and normalized.
    image_directory (str): The directory path where the image files are located.

    Returns:
    numpy.ndarray: A numpy array containing the preprocessed images.

    numpy.ndarray: A numpy array containing the target labels, i.e., the scaled bounding boxes represented as [x1_scaled, y1_scaled, x2_scaled, y2_scaled].
    """

    images = []  # List to store the preprocessed images
    labels = []  # List to store the target labels (scaled bounding boxes)

    for index, row in bbox_data.iterrows():
        image_name, x1, y1, x2, y2, _, _ = row

        # Load and normalize the image
        image = load_img(f"{image_directory}/{image_name}", target_size=target_size)
        image = img_to_array(image) / 255.0
        images.append(image)

        # Calculate scaling factors for bounding box coordinates
        original_width, original_height = load_img(
            f"{image_directory}/{image_name}"
        ).size
        scale_x = target_size[0] / original_width
        scale_y = target_size[1] / original_height

        # Resize and scale bounding box coordinates
        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y

        label = [x1_scaled, y1_scaled, x2_scaled, y2_scaled]
        labels.append(label)

    return np.array(images), np.array(labels)


def unnormalize_boxes(x1, x2, y1, y2, current_size, target_size):
    scale_x = target_size[0] / current_size[0]
    scale_y = target_size[1] / current_size[1]
    x1_unscaled = x1 * scale_x
    x2_unscaled = x2 * scale_x
    y1_unscaled = y1 * scale_y
    y2_unscaled = y2 * scale_y
    return x1_unscaled, x2_unscaled, y1_unscaled, y2_unscaled


def local_IoU(
    xmin_pred_i,
    xmax_pred_i,
    ymin_pred_i,
    ymax_pred_i,
    xmin_true_i,
    xmax_true_i,
    ymin_true_i,
    ymax_true_i,
):
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
