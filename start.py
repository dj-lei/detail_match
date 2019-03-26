import sys
sys.path.append('../')

from match import *


def start():
    """
    生成款型匹配余弦向量表
    """
    process = Process()
    process.generate_cos_vector()


def match():
    """
    匹配车源数据
    """
    process = Process()
    # process.match_car_source()
    process.match_test()


if __name__ == "__main__":
    # start()
    match()