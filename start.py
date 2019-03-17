import sys
sys.path.append('../')

from match import *


def start():
    """
    训练所有款型的预测
    """
    process = Process()
    process.generate_cos_vector()


if __name__ == "__main__":
    start()
