from match.process import *


class Process(object):

    def __init__(self):
        self.data = []
        self.result = []

    def train_all_details(self):
        """
        处理所有车型
        """
        # try:
        time1 = time.time()
        print('start project!')
        # 存储训练相关表
        # process_tables.store_train_relative_data()
        # # 特征工程
        # fe = FeatureEngineering()
        # fe.execute()
        # 训练模型
        stack = Stacking()
        stack.execute()
        time2 = time.time()
        print('cost time', time2-time1)

        # except Exception as e:
        #     db_operate.insert_valuate_error_info(e)

    def predict_all_car_source(self):
        """
        预测所有款型匹配
        """
        try:
            # 查询car_source需要预测记录
            process_tables.store_all_need_predict_data()
            # 预测品牌
            predict = Predict()
            predict.execute()
        except Exception as e:
            print(traceback.format_exc())

    def cron_predict(self):
        """
        定时预测款型匹配
        """
        try:
            # time_node = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime('%Y-%m-%d')
            # start_time_node = '2018-01-01 00:00:00'
            # end_time_node = '2018-11-21 23:59:59'
            # 查询car_source需要预测记录
            # process_tables.store_predict_data(start_time_node, end_time_node)
            # 预测品牌
            predict = Predict()
            predict.execute_cron()
        except Exception as e:
            print(traceback.format_exc())
            # db_operate.insert_valuate_error_info(e)
