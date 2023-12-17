import numpy as np

from PIL import Image
from io import BytesIO

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

# By ChatGPT
def resize_and_convert_to_bw(input_array, new_size):
    """
    Resize the input NumPy array and convert it to black and white.

    Parameters:
    - input_array (numpy.ndarray): Input array of shape (H, W, Channels).
    - new_size (tuple): New size of the image in the format (width, height).

    Returns:
    - output_image (PIL.Image.Image): Resized and black and white converted image.
    """

    # Convert NumPy array to Pillow Image
    input_image = Image.fromarray(input_array.astype(np.uint8))

    # Resize the image
    resized_image = input_image.resize(new_size)

    # Convert to black and white using the luminosity method (0.299*R + 0.587*G + 0.114*B)
    bw_image = resized_image.convert('1')

    return np.expand_dims(np.array(bw_image), -1)