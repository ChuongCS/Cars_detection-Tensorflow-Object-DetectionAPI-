import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import time
import cv2
from scipy.stats import norm
import os
import urllib
import tarfile
import glob

#matplotlib inline
#plt.style.use('ggplot')


'''MOBILNET = 'ssd_mobilenet_v1_coco_11_06_2017'
RFCN = 'rfcn_resnet101_coco_11_06_2017'
FASTER_INCEPTION = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'
INCEPTION = 'ssd_inception_v2_coco_11_06_2017'
FASTER_RCNN='faster_rcnn_resnet101_coco_11_06_2017'
MODEL_NAMES = [MOBILNET, RFCN, FASTER_INCEPTION, FASTER_RCNN, INCEPTION]

MODEL_NAMES = [INCEPTION]
MODEL_FILE_ENDING = '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
MODELS_DIR='models'
SIM_IMAGES_DIR='simulator_images'

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
model_directories = os.listdir(MODELS_DIR)
for model_name in MODEL_NAMES:
    if model_name not in model_directories:
        opener = urllib.request.URLopener()
        model_file = model_name + MODEL_FILE_ENDING
        deployed_model = os.path.join(MODELS_DIR, model_file)
        opener.retrieve(DOWNLOAD_BASE +  model_file , deployed_model)
        tar_file = tarfile.open(deployed_model)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, MODELS_DIR)'''


MODEL_NAME = 'models/ssd_inception_v2_coco_11_06_2017' # chuong ++
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb' # chuong ++

# HELPER FUNCTIONS

# Colors (one for each class)
cmap = ImageColor.colormap
COLOR_LIST = sorted([c for c in cmap.keys()])
#
# Utility funcs
#

def filter_boxes(min_score, boxes, scores, classes, categories):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if classes[i] in categories and scores[i] >= min_score:
            idxs.append(i)

    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes


def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].

    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width

    return box_coords


def draw_boxes(image, boxes, classes, thickness=4):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
        color = COLOR_LIST[class_id]
        draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)


#def ilustrate_detection(image, boxes, classes, title):
def ilustrate_detection(image, boxes, classes):
    # The current box coordinates are normalized to a range between 0 and 1.
    # This converts the coordinates actual location on the image.
    width, height = image.size
    box_coords = to_image_coords(boxes, height, width)

    # Each class with be represented by a differently colored box
    draw_boxes(image, box_coords, classes)
    fig = plt.figure(figsize=(12, 8))
    #fig.suptitle(title, fontsize=10)
    #plt.imshow(image)
    #plt.show()
    cv2.imshow(image)



# MODEL FUNCTIONS
def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        image_tensor = graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        detection_classes = graph.get_tensor_by_name('detection_classes:0')
    return graph, image_tensor, detection_boxes, detection_scores, detection_classes


def detect(detection_graph,
           image_tensor,
           detection_boxes,
           detection_scores,
           detection_classes,
           image_np,
           categories,
           runs=1):
    with tf.Session(graph=detection_graph) as sess:
        # Actual detection.

        times = np.zeros(runs)
        for i in range(runs):
            t0 = time.time()
            (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes],
                                                feed_dict={image_tensor: image_np})
            t1 = time.time()
            times[i] = (t1 - t0) * 1000

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        return boxes, scores, classes, times


# DETECT CAR


#frozen_model_paths=[graph for graph in glob.iglob(MODELS_DIR+'/**/*.pb', recursive=True)] # chuong --
#sim_image_paths= [ graph for graph in glob.iglob('simulator_images/*.jpg', recursive=True)] # chuong --
#traffic_sign_class_id = 10
traffic_car_class_id=3
#test_image=sim_image_paths[1] # chuong --
test_image=('tves_images/frame21200.jpg')

#for model in frozen_model_paths:


detection_graph, image_tensor, detection_boxes, detection_scores, detection_classes = load_graph(PATH_TO_CKPT)
image = Image.open(test_image)
image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
boxes, scores, classes, _ = detect(detection_graph,
                                  image_tensor,
                                  detection_boxes,
                                  detection_scores,
                                  detection_classes,
                                  image_np,
                                  traffic_car_class_id)
confidence_cutoff = 0.2
#traffic_lights_class_id=10
traffic_car_class_id= 3


# Filter boxes with a confidence score less than `confidence_cutoff`
boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes, [traffic_car_class_id])
#ilustrate_detection(image, boxes, classes, model.split('/')[1]) # when we test multiple models Chuong --
ilustrate_detection(image, boxes, classes) # chuong++