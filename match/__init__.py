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

# 重定位根路径
from match.conf import global_settings as gl
path = os.path.abspath(os.path.dirname(gl.__file__))
path = path.replace('conf', '')

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from match.exception.error import SqlOperateError
from match.exception.error import FeatureEngineeringError
from match.exception.error import StackingTrainError
from match.exception.error import PredictError

from match.db import db_operate
from match.db import process_tables
from match.generate.generate import Generate

from match.process.process import Process





