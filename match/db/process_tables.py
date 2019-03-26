from match.db import *


def insert_or_update_match_cos_vector(data):
    """
    插入或更新匹配向量表
    """
    try:
        db_operate.insert_or_update_match_cos_vector(data)
    except Exception:
        raise SqlOperateError(traceback.format_exc())


def insert_or_update_match_brand_name(data):
    """
    插入或更新匹配品牌查询表
    """
    try:
        db_operate.insert_or_update_match_brand_name(data)
    except Exception:
        raise SqlOperateError(traceback.format_exc())


def insert_or_update_match_word_index(data):
    """
    插入或更新匹配词向量表
    """
    try:
        db_operate.insert_or_update_match_word_index(data)
    except Exception:
        raise SqlOperateError(traceback.format_exc())


def query_car_source():
    """
    查询车源数据
    """
    try:
        data = db_operate.query_car_source()
        data.to_csv(path + '../tmp/train/wait_match.csv', index=False)
    except Exception:
        raise SqlOperateError(traceback.format_exc())



