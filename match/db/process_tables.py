from match.db import *


def store_train_relative_data():
    """
    查询训练相关数据表并存储在tmp中
    """
    try:
        open_model_detail = db_operate.query_produce_open_model_detail()
        open_model_detail = open_model_detail.rename(columns={'detail_model_slug': 'model_detail_slug'})
        open_model_detail.to_csv(path+'../tmp/train/open_model_detail.csv', index=False)
        del open_model_detail
        gc.collect()

        open_category = db_operate.query_produce_open_category()
        open_category.to_csv(path+'../tmp/train/open_category.csv', index=False)
        del open_category
        gc.collect()

        # model_detail_normal = db_operate.query_produce_model_detail_normal()
        # model_detail_normal.to_csv(path+'../tmp/train/model_detail_normal.csv', index=False)
        # del model_detail_normal
        # gc.collect()

        print('Download finish!')
    except Exception:
        raise SqlOperateError(traceback.format_exc())


def store_all_need_predict_data():
    """
    查询需要预测数据存储在tmp中
    """
    try:
        reslut = pd.DataFrame()
        start_time = str(datetime.datetime.now().year - 1) + '-01-01'
        # 查询生产成交记录
        data = db_operate.query_model_product_review_source_data(start_time)
        reslut = reslut.append(data)
        # 查询生产最近三个月在售记录
        pub_time = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
        time.sleep(0.2)
        data = db_operate.query_model_product_sale_source_data(pub_time)
        reslut = reslut.append(data)
        os.makedirs(os.path.dirname(path + '../tmp/train/predict.csv'), exist_ok=True)
        reslut.to_csv(path + '../tmp/train/predict.csv', index=False)
        print('Download finish!')
    except Exception:
        raise SqlOperateError(traceback.format_exc())


def store_predict_data(start_time, end_time):
    """
    查询需要预测数据存储在tmp中
    """
    try:
        # 查询生产新发布数据
        data = db_operate.query_model_product_source_data(start_time, end_time)
        # 查询爬虫更新过的数据
        # data = data.append(db_operate.query_spider_update_product_source_data(start_time, end_time))

        os.makedirs(os.path.dirname(path + '../tmp/train/wait_predict.csv'), exist_ok=True)
        data.to_csv(path + '../tmp/train/wait_predict.csv', index=False)
        print('Download finish!')
    except Exception:
        raise SqlOperateError(traceback.format_exc())





