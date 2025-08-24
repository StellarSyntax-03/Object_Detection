# **Importing necessary libraries**
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
from seaborn import color_palette
import cv2

# **Model hyperparameters definition**
_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1
_ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
_MODEL_SIZE = (416, 416)

# **`batch_norm` function**
def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        scale=True, training=training)

# **`fixed_padding` function**
def fixed_padding(inputs, kernel_size, data_format):
    """ResNet implementation of fixed padding. Pads the input along the spatial dimensions independently of input size.
    Args:
    inputs: Tensor input to be padded.
    kernel_size: The kernel to be used in the conv2d or max_pool2d.
    data_format: The input format.
    Returns:
    A tensor with the same format as the input.
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [,, [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [, [pad_beg, pad_end], [pad_beg, pad_end],])
    return padded_inputs

# **`conv2d_fixed_padding` function**
def conv2d_fixed_padding(inputs, filters, kernel_size, data_format, strides=1):
    """Strided 2-D convolution with explicit padding."""
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size,
        strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False, data_format=data_format)

# **`darknet53_residual_block` function**
def darknet53_residual_block(inputs, filters, training, data_format, strides=1):
    """Creates a residual block for Darknet."""
    shortcut = inputs
    inputs = conv2d_fixed_padding(
        inputs, filters=filters, kernel_size=1, strides=strides,
        data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    inputs = conv2d_fixed_padding(
        inputs, filters=2 * filters, kernel_size=3, strides=strides,
        data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    inputs += shortcut
    return inputs

# **`darknet53` function**
def darknet53(inputs, training, data_format):
    """Creates Darknet53 model for feature extraction."""
    inputs = conv2d_fixed_padding(inputs, filters=32, kernel_size=3,
                                data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    inputs = conv2d_fixed_padding(inputs, filters=64, kernel_size=3,
                                strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    inputs = darknet53_residual_block(inputs, filters=32, training=training,
                                    data_format=data_format)
    inputs = conv2d_fixed_padding(inputs, filters=128, kernel_size=3,
                                strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    for _ in range(2):
        inputs = darknet53_residual_block(inputs, filters=64,
                                        training=training,
                                        data_format=data_format)
    inputs = conv2d_fixed_padding(inputs, filters=256, kernel_size=3,
                                strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=128,
                                        training=training,
                                        data_format=data_format)
    route1 = inputs
    inputs = conv2d_fixed_padding(inputs, filters=512, kernel_size=3,
                                strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=256,
                                        training=training,
                                        data_format=data_format)
    route2 = inputs
    inputs = conv2d_fixed_padding(inputs, filters=1024, kernel_size=3,
                                strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    for _ in range(4):
        inputs = darknet53_residual_block(inputs, filters=512,
                                        training=training,
                                        data_format=data_format)
    return route1, route2, inputs

# **`yolo_convolution_block` function**
def yolo_convolution_block(inputs, filters, training, data_format):
    """Creates convolution operations layer used after Darknet."""
    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    inputs = conv2d_fixed_padding(inputs, filters=2 * filters, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)

# **`upsample` function**
def upsample(inputs, out_shape, data_format):
    """Upsamples to `out_shape` using nearest neighbor interpolation."""
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs,)
        new_height = out_shape
        new_width = out_shape
    else:
        new_height = out_shape
        new_width = out_shape
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs,)
    return inputs

# **`load_images` function**
def load_images(img_names, model_size):
    """Loads images in a 4D array.
    Args:
    img_names: A list of images names.
    model_size: The input size of the model.
    data_format: A format for the array returned
    ('channels_first' or 'channels_last').
    Returns:
    A 4D NumPy array.
    """
    imgs = []
    for img_name in img_names:
        img = Image.open(img_name)
        img = img.resize(size=model_size)
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        imgs.append(img)
    imgs = np.concatenate(imgs)
    return imgs

# **`load_class_names` function**
def load_class_names(file_name):
    """Returns a list of class names read from `file_name`."""
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

# **`draw_boxes` function**
def draw_boxes(img_names, boxes_dicts, class_names, model_size):
    """Draws detected boxes.
    Args:
    img_names: A list of input images names.
    boxes_dict: A class-to-boxes dictionary.
    class_names: A class names list.
    model_size: The input size of the model.
    Returns:
    None.
    """
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    for num, img_name, boxes_dict in zip(range(len(img_names)), img_names, boxes_dicts):
        img = Image.open(img_name)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font='futur.ttf', size=(img.size + img.size) // 100)
        resize_factor = (img.size / model_size, img.size / model_size)
        for cls in range(len(class_names)):
            boxes = boxes_dict[cls]
            if np.size(boxes) != 0:
                color = colors[cls]
                for box in boxes:
                    xy, confidence = box[:4], box
                    xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
                    x0, y0 = xy, xy
                    thickness = (img.size + img.size) // 200
                    for t in np.linspace(0, 1, thickness):
                        xy, xy = xy + t, xy + t
                        xy, xy = xy - t, xy - t
                    draw.rectangle(xy, outline=tuple(color))
                    text = '{} {:.1f}%'.format(class_names[cls], confidence * 100)
                    text_size = draw.textsize(text, font=font)
                    draw.rectangle(
                        [x0, y0 - text_size, x0 + text_size, y0],
                        fill=tuple(color))
                    draw.text((x0, y0 - text_size), text, fill='black', font=font)
        display(img)

# **`load_weights` function**
def load_weights(variables, file_name):
    """Reshapes and loads official pretrained Yolo weights.
    Args:
    variables: A list of tf.Variable to be assigned.
    file_name: A name of a file containing weights.
    Returns: A list of assign operations.
    """
    with open(file_name, "rb") as f: # Skip first 5 values