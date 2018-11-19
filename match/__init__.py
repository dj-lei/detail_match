import os
import gc
import re
import sys
import time
import random
import datetime
import traceback
import pickle
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
# 只显示 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# 重定位根路径
from match.conf import algorithm_settings as als
from match.conf import global_settings as gl
path = os.path.abspath(os.path.dirname(gl.__file__))
path = path.replace('conf', '')

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')

import h5py
from numpy import argmax
from sqlalchemy import create_engine
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from match.exception.error import SqlOperateError
from match.exception.error import FeatureEngineeringError
from match.exception.error import StackingTrainError
from match.exception.error import PredictError

from match.db import db_operate
from match.db import process_tables
from match.train.feature_engineering import FeatureEngineering
from match.train.stacking import Stacking
from match.predict.predict import Predict

from match.process.process import Process





