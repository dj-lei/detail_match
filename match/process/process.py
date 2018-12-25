from match.process import *


class Process(object):

    def __init__(self):
        self.data = []
        self.result = []

    def train(self):
        """
        处理所有车型
        """
        # try:
        time1 = time.time()
        print('start project!')
        # 存储训练相关表
        process_tables.store_train_relative_data()
        # 特征工程
        fe = FeatureEngineering()
        fe.execute()
        # 训练模型
        # stack = Stacking()
        # stack.execute()
        time2 = time.time()
        print('cost time', time2-time1)

    def predict(self):
        """
        定时预测款型匹配
        """
        try:
            # time_node = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime('%Y-%m-%d')
            # start_time_node = '2018-06-01 00:00:00'
            # end_time_node = '2018-11-21 23:59:59'
            # # 查询car_source需要预测记录
            # process_tables.store_predict_data(start_time_node, end_time_node)
            # 预测品牌
            predict = Predict()
            predict.execute()
        except Exception as e:
            print(traceback.format_exc())
            # db_operate.insert_valuate_error_info(e)
