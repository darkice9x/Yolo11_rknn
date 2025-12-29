
import cv2
import numpy as np

name = "YOLO"
def print_info(messages: str):
    print( f"{name} : {messages}")

def letterbox(image, target_width, target_height, bg_color):
    if isinstance(image, str):
        image = cv2.imread(image)

    if image is None:
        raise ValueError("Input image is None")

    image_height, image_width = image.shape[:2]

    aspect_ratio = min(target_width / image_width,
                       target_height / image_height)

    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    image = cv2.resize(image, (new_width, new_height),
                       interpolation=cv2.INTER_AREA)

    result_image = np.ones(
        (target_height, target_width, 3),
        dtype=np.uint8
    ) * bg_color

    result_image[0:new_height, 0:new_width] = image
    return result_image, aspect_ratio
