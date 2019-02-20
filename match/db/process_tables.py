from match.db import *


def store_train_relative_data():
    """
    查询训练相关数据表并存储在tmp中
    """
    try:
        crawler_car_source = db_operate.query_produce_crawler_car_source()
        crawler_car_source.to_csv(path+'../tmp/train/origin_train.csv', index=False)

        print('Download finish!')
    except Exception:
        raise SqlOperateError(traceback.format_exc())





