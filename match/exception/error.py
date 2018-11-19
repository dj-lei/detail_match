from match.exception import *


class SqlOperateError(Exception):
    """
    操作数据库异常
    """
    def __init__(self, message):
        self.error_type = gl.ERROR_SQL
        self.message = message


class FeatureEngineeringError(Exception):
    """
    车型特征工程异常
    """
    def __init__(self, message):
        self.error_type = gl.ERROR_FE
        self.message = message


class StackingTrainError(Exception):
    """
    车型训练异常
    """
    def __init__(self, message):
        self.error_type = gl.ERROR_TRAIN
        self.message = message


class PredictError(Exception):
    """
    车型训练异常
    """
    def __init__(self, message):
        self.error_type = gl.ERROR_PREDICT_MODEL
        self.message = message


