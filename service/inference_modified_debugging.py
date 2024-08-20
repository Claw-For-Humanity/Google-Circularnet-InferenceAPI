# conda create -n circularnet python=3.10.12
# pre-requisites: pip install tensorflow
# pre-requisites:  pip install tf_keras 
# matplotlib
# pillow
# pandas
# scikit-image
# pip install scikit-learn
# pip install webcolors
# pip install tensorflow-object-detection-api

# also for the visualization utils (/home/{username}/.local/lib/python{version}/site-packages/object_detection/utils/visualization_utils.py):
# 'display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]'
# into 'display_str_heights = [font.getbbox(ds)[3] - font.getbbox(ds)[1] for ds in display_str_list]'
# &&
# 'text_width, text_height = font.getsize(display_str)' into 'text_width, text_height = font.getbbox(display_str)[2] - font.getbbox(display_str)[0], font.getbbox(display_str)[3] - font.getbbox(display_str)[1]'


import time
start_time = time.perf_counter()
print('import beginning')

from six.moves.urllib.request import urlopen
from six import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np
import sys, os
import logging
import pandas as pd
import subprocess
logging.disable(logging.WARNING)
# %matplotlib inline 
sys.path.append(os.path.join(os.getcwd(),'service','tensorflow','research'))
from object_detection.utils import visualization_utils as viz_utils
sys.path.append(os.path.join(os.getcwd(),'service','tensorflow','official','projects','waste_identification_ml','model_inference'))
import preprocessing
import postprocessing
import color_and_property_extractor
import labels

from typing import Dict, Any
print(f'import done. took {time.perf_counter()-start_time} s\n')


# if gpu exists

gpu_devices = tf.config.experimental.list_physical_devices('GPU')

for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)




class bucket:
  detection_fns = None
  category_indices, category_index = None, None

  current_ws = os.getcwd().replace("\\",'/')

  # Path to a sample image stored in the repo.
  IMAGES_FOR_TEST = f'{current_ws}/sample_images/image_2.png'

  IMAGES_FOR_TEST1 = f'{current_ws}/sample_images/image_3.jpg'

  IMAGES_FOR_TEST2 = f'{current_ws}/sample_images/image_4.png'

  IMAGES_FOR_TEST3 = f'{current_ws}/sample_images/image_2.png'



class utils:
  def data_manager(data):
    # convert everything into list
    data = {key: value.tolist() for key, value in data.items()}
    detection_data = {}
    for i in range(data['num_detections'][0]):
        detection_data[i] = {
            'score': data['detection_scores'][0][i],
            'box': data['detection_boxes'][0][i],
            'class_name': data['detection_classes_names'][i]
        }
    return detection_data

  def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (1, h, w, 3)
    """
    image = None
    if(path.startswith('http')):
      response = urlopen(path)
      image_data = response.read()
      image_data = BytesIO(image_data)
      image = Image.open(image_data)
    else:
      image_data = tf.io.gfile.GFile(path, 'rb').read()
      image = Image.open(BytesIO(image_data))

    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (1, im_height, im_width, 3)).astype(np.uint8)


  def load_model(model_handle):
      """Loads a TensorFlow SavedModel and returns a function that can be used to make predictions.

      Args:
        model_handle: A path to a TensorFlow SavedModel.

      Returns:
        A function that can be used to make predictions.
      """
      print('loading model...')
      print(model_handle)
      model = tf.saved_model.load(model_handle)
      print('model loaded!')
      detection_fn = model.signatures['serving_default']
      return detection_fn



class initialize:
  if os.name == 'nt':
    current_ws = os.getcwd().replace("\\",'/')

  else:
    current_ws = os.getcwd()

  MODELS_RESNET_V1 = {
  'material_model' : f'{current_ws}/service/weights/resnet_material_v1/saved_model/',
  'material_form_model' : f'{current_ws}/service/weights/resnet_material_form_v1/saved_model/',
  }

  MODELS_RESNET_V2 = {
  'material_model' : f'{current_ws}/service/weights/resnet_material_v2/saved_model/',
  'material_form_model' : f'{current_ws}/service/weights/resnet_material_form_v2/saved_model/',
  }

  MODELS_MOBILENET_V2 = {
  'material_model' : f'{current_ws}/service/weights/mobilenet_material/saved_model/',
  'material_form_model' : f'{current_ws}/service/weights/mobilenet_material_form/saved_model/',
  }

  LABELS = {
      'material_model': (
          f'{current_ws}/service/tensorflow/official/projects/waste_identification_ml/pre_processing/'
          'config/data/two_model_strategy_material.csv'
      ),
      'material_form_model': (
          f'{current_ws}/service/tensorflow/official/projects/waste_identification_ml/pre_processing/'
          'config/data/two_model_strategy_material_form.csv'
      ),
  }
  
  
  # Import pre-trained models.
  selected_model = "MODELS_WEIGHTS_RESNET_V2"

  if selected_model == "MODELS_WEIGHTS_RESNET_V1":
    ALL_MODELS = MODELS_RESNET_V1
  elif selected_model == "MODELS_WEIGHTS_RESNET_V2":
    ALL_MODELS = MODELS_RESNET_V2
  elif selected_model == "MODELS_WEIGHTS_MOBILENET_V2":
    ALL_MODELS = MODELS_MOBILENET_V2


  bucket.category_indices, bucket.category_index = labels.load_labels(LABELS)

  path_existence = [os.path.isdir(path) for path in ALL_MODELS.values()]

  if False in path_existence:
    print(path_existence)
    print('error, no weights found.')
    exit()

  # Loading both models.
  bucket.detection_fns = [utils.load_model(model_path) for model_path in ALL_MODELS.values()]



class main:
  def perform_detection(model, image):
    """Performs Mask RCNN on an image using the specified model.

    Args:
      model: A function that can be used to make predictions.
      image_np: A NumPy array representing the image to be detected.

    Returns:
      A list of detections.
    """
    detection_fn = model(image)
    detection_fn = {key: value.numpy() for key, value in detection_fn.items()}
    return detection_fn


  def load_img(image_path):
    image_np = utils.load_image_into_numpy_array(str(image_path))

    # print('min:', np.min(image_np[0]), 'max:', np.max(image_np[0]))

    print('image is loaded\n')

    return image_np


  def pre_processing(image_np):
    height = bucket.detection_fns[0].structured_input_signature[1]['inputs'].shape[1]
    width = bucket.detection_fns[0].structured_input_signature[1]['inputs'].shape[2]
    input_size = (height, width)
    print(f'input size is {height}x{width}')

    image_np_cp = tf.image.resize(image_np[0], input_size, method=tf.image.ResizeMethod.AREA)
    image_np_cp = tf.cast(image_np_cp, tf.uint8)
    image_np = preprocessing.normalize_image(image_np_cp)
    image_np = tf.expand_dims(image_np, axis=0)
    image_np.get_shape()

    print('pre processing done\n')


    return image_np, image_np_cp, height, width


  def display_final_result(final_result, image_np_cp):
    image_new = image_np_cp.numpy().copy()

    if 'detection_masks_reframed' in final_result:
      final_result['detection_masks_reframed'] = final_result['detection_masks_reframed'].astype(np.uint8)

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_new,
          final_result['detection_boxes'][0],
          (final_result['detection_classes'] + 0).astype(int),
          final_result['detection_scores'][0],
          category_index=bucket.category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=100,
          min_score_thresh=0.6,
          agnostic_mode=False,
          instance_masks=final_result.get('detection_masks_reframed', None),
          line_thickness=2)
    
    return image_new

  def get_detection_info(final_result):
      num_objects = int(final_result['num_detections'][0])
      object_types = final_result['detection_classes_names']
      detection_boxes = final_result['detection_boxes'][0]

      coordinates = []
      for box in detection_boxes:
          ymin, xmin, ymax, xmax = box
          x_center = (xmin + xmax) / 2
          y_center = (ymin + ymax) / 2
          coordinates.append((x_center, y_center))

      return num_objects, object_types, coordinates




  def inference(np_img = None, is_display=False, min_confidence = 0.8):
    
    image_np, image_np_cp, height, width = main.pre_processing(np_img)

    print('performing inference')
    s_i = time.perf_counter()
    results = list(map(lambda model: main.perform_detection(model, image_np), bucket.detection_fns))
    print(f'inference took {time.perf_counter()-s_i}\n')

    no_detections_in_first = results[0]['num_detections'][0]
    no_detections_in_second = results[1]['num_detections'][0]

    if no_detections_in_first !=0 and no_detections_in_second != 0:
      results = [postprocessing.reframing_masks(detection, height, width) for detection in results]

      max_detection = max(no_detections_in_first, no_detections_in_second)

      area_threshold = 0.3 * np.prod(image_np_cp.shape[:2])

      final_result = postprocessing.find_similar_masks(
          results[0],
          results[1],
          max_detection,
          min_confidence,
          bucket.category_indices,
          bucket.category_index,
          area_threshold
      )

      transformed_boxes = []
      for bb in final_result['detection_boxes'][0]:
          YMIN = int(bb[0]*height)
          XMIN = int(bb[1]*width)
          YMAX = int(bb[2]*height)
          XMAX = int(bb[3]*width)
          transformed_boxes.append([YMIN, XMIN, YMAX, XMAX])

      filtered_boxes, index_to_delete = (
        postprocessing.filter_bounding_boxes(transformed_boxes))

      final_result['num_detections'][0] -= len(index_to_delete)
      final_result['detection_classes'] = np.delete(
          final_result['detection_classes'], index_to_delete)
      final_result['detection_scores'] = np.delete(
          final_result['detection_scores'], index_to_delete, axis=1)
      final_result['detection_boxes'] = np.delete(
          final_result['detection_boxes'], index_to_delete, axis=1)
      final_result['detection_classes_names'] = np.delete(
          final_result['detection_classes_names'], index_to_delete)
      final_result['detection_masks_reframed'] = np.delete(
          final_result['detection_masks_reframed'], index_to_delete, axis=0)

      if final_result != None:
        if is_display:
          image_np_cp = main.display_final_result(final_result, image_np_cp)
      
        final_result.pop('detection_classes')
        final_result.pop('detection_masks_reframed')
        
        final_result = utils.data_manager(final_result)      
        
        
      else:
        print('\nnothing detected\n')
        final_result, image_np_cp = None, None
    

    return final_result, image_np_cp
  

# initialize
# i = time.perf_counter()


# img = main.load_img(bucket.IMAGES_FOR_TEST2)

# final_result, image_np_cp = main.inference(img, is_display=False)
# print(f'took {time.perf_counter()-i} s')

# print(f'{final_result}\n')