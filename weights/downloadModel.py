
from tensorflow.keras import applications
import os

if __name__ == "__main__":
    
    model = applications.ResNet50(weights='imagenet')
    print(model.summary())
    model.save('resNet50_imagenet.hdf5')
    