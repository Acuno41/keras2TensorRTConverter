# keras2TensorRTConverter

keras2TensorRTConverter is a Python program that allow you to convert keras .hdf5 models to optimized tensorflow .pb files.

**Tested on Ubuntu 18.04
Tensorflow 1.14**

## Download 

git clone https://github.com/Acuno41/keras2TensorRTConverter.git

## Installation

Use the [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip3 install requirements.txt
```

## Usage
Download Keras pre-trained resNet50 imagenet model with:
```bash
python3 weights/downloadModel.py
```
To convert .hdf5 file to optimized .pb file run:
```bash
python3 keras2TensorRTConverter.py --modelDir weights/ --modelName resNet50_imagenet.hdf5
```
to make inference with the optimized tensorrt model run:
(The important point should be paid attention to the **input and output layer names in the previous script output**.)
```bash
python3 predictTensorrt.py --modelPath weights/resNet50_imagenet.pb --imagePath images/container.jpeg --inputLayerName input_1 --outputLayerName fc1000/Softmax
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.