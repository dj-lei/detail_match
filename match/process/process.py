from match.process import *


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
        # process_tables.query_car_source()

        # 匹配
        result = pd.DataFrame()
        car_source = pd.read_csv(path + '../tmp/train/wait_match.csv')
        detail_name = car_source.loc[:, ['detail_name']].sort_values(by=['detail_name'])
        detail_name = detail_name.drop_duplicates(['detail_name']).reset_index(drop=True)

        print('等待匹配车源数据量:', len(car_source), '款型数量:', len(detail_name))
        match = Match()
        for i, detail in enumerate(list(detail_name.loc[:, 'detail_name'].values)):

            part1 = car_source.loc[(car_source['detail_name'] == detail), :].reset_index(drop=True)
            print(i, detail_name['detail_name'][i], len(part1))
            gpj_detail = match.predict(part1['detail_name'][0])['data']
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
                                'energy', 'body', 'control', 'volume', 'year', 'month', 'mile', 'city', 'price_bn', 'price', 'create_time', 'domain', 'labels', 'url']]
        result.to_csv(path + '../tmp/train/train_temp.csv', index=False)

    def match_test(self):
        """
        简单测试
        """
        match = Match()
        gpj_detail = match.predict('雪佛兰 乐驰 2006款 0.8L 手动舒适型', cos_similar=0)
        print(gpj_detail)
        return gpj_detail