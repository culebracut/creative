from __future__ import absolute_import, division, print_function

import functools
import os
from datetime import datetime

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import os, random

#outputPath = '/media/19F1-354C/data/output/'
#imagePath = '/media/19F1-354C/data/images/'
#modelPath = '/media/19F1-354C/data/arbitrary-image-stylization-v1-256/'
outputPath = '/media/system/UBUNTU 18_0/data/output/'
#imagePath = '/media/system/UBUNTU 18_0/data/images/'
imagePath = '/media/system/UBUNTU 18_0/art/resized/resized/'
styleImagePath = '/media/system/UBUNTU 18_0/art/resized/resized/'
modelPath = '/media/system/UBUNTU 18_0/data/arbitrary-image-stylization-v1-256/'

print("TF Version: ", tf.__version__)
print("TF-Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.test.is_gpu_available())

# Load TF-Hub module.
#modelPath = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
print('loading model...{}'.format(modelPath))
hub_module = hub.load(modelPath)
#tf.saved_model.save(hub_module, "../data/")
print('model loaded')

################################################################################3
# helper image functions
# @title Define image loading and visualization functions  { display-mode: "form" }
def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  # TW: modified to load disk files
  image_path = tf.keras.utils.get_file(image_url, image_url)

  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
  if img.max() > 1.0:
    img = img / 255.
  if len(img.shape) == 3:
    img = tf.stack([img, img, img], axis=-1)
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

def show_n(images, titles=('',)):
  n = len(images)
  image_sizes = [image.shape[1] for image in images]
  w = (image_sizes[0] * 6) // 320
  plt.figure(figsize=(w  * n, w))
  gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
  for i in range(n):
    plt.subplot(gs[i])
    plt.imshow(images[i][0], aspect='equal')
    plt.axis('off')
    plt.title(titles[i] if len(titles) > i else '')
  plt.show()
  #plt.savefig(outputPath+datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p"))

###############################################################################
for i in range(10):
    # Get files
    style_image_url = styleImagePath + random.choice(os.listdir(styleImagePath)) 
    content_image_url = imagePath + random.choice(os.listdir(imagePath)) 
    print (style_image_url)
    print (content_image_url)

    # The content image size can be arbitrary.
    output_image_size = 384  # @param {type:"integer"}
    content_img_size = (output_image_size, output_image_size)
    style_img_size = (256, 256)  # Recommended to keep it at 256.
    #
    # The style prediction model was trained with image size 256 and it's the 
    # recommended image size for the style image (though, other sizes work as 
    # well but will lead to different results).
    content_image = load_image(content_image_url, content_img_size)
    style_image = load_image(style_image_url, style_img_size)
    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
    #show_n([content_image, style_image], ['Content image', 'Style image'])

    # Stylize image.
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]

    # Visualize input images and the generated stylized image.
    show_n([content_image, style_image, stylized_image], titles=[os.path.basename(content_image_url), os.path.basename(style_image_url), 'Stylized image'])
