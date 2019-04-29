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
    result = process.match_test('上汽大通 D90 2017款 2.0T 手动 两驱 火星')
    print(result)
    # print(result['data']['origin_name'],'|',result['data']['brand_name'],'|',result['data']['model_name'],'|',result['data']['detail_name'],'|',result['data']['cos_similar'])


def match_old():
    """
    旧版纯字符串匹配
    """
    process = Process()
    process.match_old_version()


def sync_details():
    """
    同步生产款型库
    """
    process = Process()
    process.sync_details()


if __name__ == "__main__":
    start()
    # match()
    # sync_details()