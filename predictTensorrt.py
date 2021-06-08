from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import numpy as np
import argparse
import time
import cv2
import sys

# example usage --> python predictTensorrt.py --modelPath weights/resNet50_imagenet.pb --imagePath images/container.jpeg --inputLayerName input_1 --outputLayerName fc1000/Softmax

def parseArgs():
    parser = argparse.ArgumentParser(description='Image inference')
    parser.add_argument(
        '--inputLayerName',
        dest='inputLayerName',
        help='Name of the models input layer',
        default='input_1',
        type=str
    )
    parser.add_argument(
        '--outputLayerName',
        dest='outputLayerName',
        help='Name of the models output layer',
        default='fc1000/Softmax',
        type=str
    )
    parser.add_argument(
        '--modelPath',
        dest='modelPath',
        help='Path of the converted .pb file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--imagePath',
        dest='imagePath',
        help='Path to image',
        default=None,
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def getFrozenGraph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def getGraphInputSize(trt_graph, input_names):
    for node in trt_graph.node:
        if input_names[0] in node.name:
            size = node.attr['shape'].shape
            imageSize = [size.dim[i].size for i in range(1, 4)]
            break
    print("image_size: {}".format(imageSize))
    return imageSize

def readImage(imagePath, image_size):
    im = cv2.imread(imagePath)   
    im = cv2.resize(im,(image_size[0], image_size[1]))
    imCopy = im.copy() 
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    return im, imCopy


if __name__ == "__main__":

    K.set_learning_phase(0) # to increase pred speed
    args = parseArgs()

    inputNames = [args.inputLayerName]
    outputNames = [args.outputLayerName]

    trtGraph = getFrozenGraph(args.modelPath)

    # Create session and load graph
    tfConfig = tf.ConfigProto()
    tfConfig.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tfConfig)
    tf.import_graph_def(trtGraph, name='')

    # Get graph input size
    image_size = getGraphInputSize(trtGraph, inputNames)  

    inputTensorName = inputNames[0] + ":0"
    outputTensorName = outputNames[0] + ":0"
    outputTensor = tf_sess.graph.get_tensor_by_name(outputTensorName)

    # Read image and prepare for model input
    img, copy = readImage(args.imagePath, image_size)

    # Run pred on image
    feedDict = {inputTensorName: img }
    prediction = tf_sess.run(outputTensor, feedDict)
    
    cv2.putText(copy,'ClassId: ' + str(np.argmax(prediction)), (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(copy,'Score: %' + str(round(np.amax(prediction),3)) ,(5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)

    cv2.imshow('image',copy)
    cv2.waitKey(0)
    