from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import argparse
import sys

# example usage --> python keras2TensorRTConverter.py --modelDir weights/ --modelName resNet50_imagenet.hdf5
# Download Keras pre-trained model with weights/downloadModel.py
def parseArgs():
    parser = argparse.ArgumentParser(description='Image inference')
    parser.add_argument(
        '--modelDir',
        dest='modelDir',
        help='Keras hdf5 model dir(/path/to/model.hdf5)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--modelName',
        dest='modelName',
        help='name of the hdf5 file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--precisionMode',
        dest='precisionMode',
        help='Precision Mode (INT8, FP16, FP32)',
        default='FP16',
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen


if __name__ == "__main__":

    args = parseArgs()
    
    # Clear any previous session.
    tf.keras.backend.clear_session()
    # This line must be executed before loading Keras model.
    tf.keras.backend.set_learning_phase(0) 

    # Full path of .hdf5 file
    modelFname = args.modelDir + args.modelName
    
    # Load keras .hdf5 model
    model = load_model(modelFname)
    session = tf.keras.backend.get_session()

    # Get input and output layers name 
    inputNames = [t.op.name for t in model.inputs]
    outputNames = [t.op.name for t in model.outputs]
    # Prints input and output nodes names, take notes of them. They will be needed in inference
    print('\nInput Node Name = ',inputNames, '\nutput Node Name = ', outputNames)

    # Freeze model
    frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=args.modelDir)

    # Optimize model with tensorRT conversation
    trt_graph = trt.create_inference_graph(
                    input_graph_def=frozen_graph,
                    outputs=outputNames,
                    max_batch_size=1,
                    max_workspace_size_bytes=1 << 25,
                    precision_mode=args.precisionMode,
                    minimum_segment_size=50
                    )

    # Save optimized model 
    graph_io.write_graph(trt_graph, args.modelDir, args.modelName[:-4]+'pb', as_text=False)
    