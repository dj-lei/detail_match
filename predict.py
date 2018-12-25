import sys
sys.path.append('../')

from match import *


def cron_predict():
    """
    训练所有款型的预测
    """
    process = Process()
    process.cron_predict()


if __name__ == "__main__":
    cron_predict()
