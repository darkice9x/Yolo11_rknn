
import cv2
import numpy as np

name = "YOLO"
def print_info(messages: str):
    print( f"{name} : {messages}")

def letterbox(image, target_width, target_height, bg_color):
    """
    letterbox the image according to the specified size
    :param image: input image, which can be a NumPy array or file path
    :param size: target size (width, height)
    :param bg_color: background filling data 
    :return: processed image
    """
    if isinstance(image, str):
        image = cv2.imread(image)

    #target_width, target_height = input_size, input_size
    image_height, image_width, _ = image.shape

    # Calculate the adjusted image size
    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    # Use cv2.resize() for proportional scaling
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new canvas and fill it
    result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
    #offset_x = (target_width - new_width) // 2
    #offset_y = (target_height - new_height) // 2
    result_image[0:new_height, 0:new_width] = image
    return result_image, aspect_ratio