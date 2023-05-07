import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
import tensorflow as tf
from glob import glob

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model


from tensorflow.keras.applications.resnet_v2 import ResNet152V2,preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential


from pyswarms.single.global_best import GlobalBestPSO
from tensorflow.keras.callbacks import EarlyStopping

