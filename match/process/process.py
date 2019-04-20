from match.process import *


def timestamp_datetime(df):
    deal_date = int(df['deal_date_ts'])
    return datetime.datetime.fromtimestamp(deal_date).strftime("%Y-%m-%d")


class Process(object):

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

    def match_car_source(self):
        """
        匹配车源数据
        """
        # 查询车源数据
        process_tables.query_car_source()

        # 匹配
        print('start project!')
        result = pd.DataFrame()
        car_source = pd.read_csv(path + '../tmp/train/wait_match.csv', low_memory=False)

        ttpai = car_source.loc[(car_source['domain'] == 'ttpai.cn'), :].reset_index(drop=True)
        ttpai['detail_name'] = ttpai['brand_name'] + ' ' + ttpai['model_name'] + ' ' + ttpai['detail_name']
        ttpai['sold_time'] = ttpai.apply(timestamp_datetime, axis=1)

        others = car_source.loc[(car_source['domain'] != 'ttpai.cn'), :].reset_index(drop=True)
        car_source = others.append(ttpai, sort=False)
        car_source = car_source.loc[(car_source['detail_name'].notnull()), :].reset_index(drop=True)
        car_source = car_source.drop(['brand_name', 'model_name'], axis=1)

        detail_name = car_source.loc[:, ['detail_name']].sort_values(by=['detail_name'])
        detail_name = detail_name.drop_duplicates(['detail_name']).reset_index(drop=True)

        print('等待匹配车源数据量:', len(car_source), '款型数量:', len(detail_name))
        match = Match()
        for i, detail in enumerate(list(detail_name.loc[:, 'detail_name'].values)):

            part1 = car_source.loc[(car_source['detail_name'] == detail), :].reset_index(drop=True)
            print(i, detail_name['detail_name'][i], len(part1))
            gpj_detail = match.predict(part1['detail_name'][0], cos_similar=0.81)['data']
            print(gpj_detail)
            if len(gpj_detail) == 0:
                part1['origin_name'] = part1['detail_name']
                part1 = part1.drop(['detail_name'], axis=1)
                result = result.append(part1, sort=False).reset_index(drop=True)
            else:
                part2 = pd.DataFrame(gpj_detail, index=[0])
                part1['origin_name'] = part1['detail_name']
                part1 = part1.drop(['detail_name'], axis=1)
                temp = pd.concat([part1, pd.DataFrame(columns=list(part2.columns))], sort=False)
                temp.loc[:, part2.columns] = part2.loc[0, :].values
                result = result.append(temp, sort=False).reset_index(drop=True)

        result = result.loc[:, ['origin_name', 'brand_name', 'model_name', 'detail_name', 'cos_similar', 'brand_slug', 'model_slug', 'detail_slug', 'online_year',
                                'energy', 'body', 'control', 'volume', 'year', 'month', 'mile', 'city', 'price_bn', 'price', 'create_time', 'domain', 'labels', 'url', 'sold_time']]
        result.to_csv(path + '../tmp/train/train_temp.csv', index=False)

        # 生成训练数据
        train_temp = pd.read_csv(path + '../tmp/train/train_temp.csv')

        start_time = datetime.datetime.now() - datetime.timedelta(days=60)
        start_time = start_time.strftime("%Y-%m-%d") + ' 00:00:00'

        ttpai = train_temp.loc[(train_temp['domain'] == 'ttpai.cn'), :].reset_index(drop=True)
        ttpai = ttpai.loc[(ttpai['sold_time'] >= start_time), :].reset_index(drop=True)
        ttpai['type'] = 'sell'
        others = train_temp.loc[(train_temp['domain'] != 'ttpai.cn'), :].reset_index(drop=True)
        others['type'] = 'personal'

        train_temp = others.append(ttpai, sort=False)
        train_temp = train_temp.loc[(train_temp['brand_name'].notnull()) & (train_temp['year'].notnull()),:].reset_index(drop=True)
        train_temp.to_csv('/home/ml/ProgramProject/evaluation-predict/tmp/train/train.csv', index=False)

        # 存储未匹配上车源
        miss_match = train_temp.loc[(train_temp['brand_name'].isnull()), :].reset_index(drop=True)
        if os.path.exists(path + '../tmp/train/miss_match.csv'):
            miss_match.to_csv(path + '../tmp/train/miss_match.csv', index=False)
        else:
            miss_match.to_csv(path + '../tmp/train/miss_match.csv', index=False, mode='a', header=False)

    def match_test(self, detail_name):
        """
        简单测试
        """
        match = Match()
        gpj_detail = match.predict(detail_name, cos_similar=0)
        return gpj_detail

    def match_old_version(self):
        """
        旧版纯字符串匹配
        """
        feature = FeatureEngineering()
        feature.execute()