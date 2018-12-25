import sys
sys.path.append('../')

from match import *


def train():
    """
    训练所有款型的预测
    """
    process = Process()
    process.train()


if __name__ == "__main__":
    train()
