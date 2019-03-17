from match.process import *


class Process(object):

    def __init__(self):
        pass

    def generate_cos_vector(self):
        """
        生成款型余弦向量
        """
        # try:
        time1 = time.time()
        print('start project!')
        generate = Generate()
        generate.execute()
        time2 = time.time()
        print('cost time', time2-time1)

