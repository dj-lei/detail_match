import sys
sys.path.append('../')

from match import *


def train_all_details():
    """
    训练所有款型的预测
    """
    process = Process()
    process.train_all_details()


if __name__ == "__main__":
    train_all_details()
