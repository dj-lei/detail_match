from match.predict import *


def process_brand_name(df):
    """
    删除没用的字符
    """
    text = df['brand_name']

    text = text.replace('Jeep', '吉普')
    text = text.replace('SWM斯威汽车', '斯威')
    text = text.replace('名爵', 'MG')
    text = text.replace('上汽大通', '大通')
    text = text.replace('福汽启腾', '启腾')
    text = text.replace('汽车', '')
    text = text.replace('・', '')
    text = text.replace(' ', '')
    text = text.lower()
    return text


def process_model_name(df):
    """
    删除没用的字符
    """
    text = df['model_name']
    text = text.lower()
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('（', '')
    text = text.replace('）', '')
    text = text.replace('-', '')
    text = text.replace('·', '')
    text = text.replace('・', '')
    text = text.replace('/', '')
    text = text.replace('°', '')
    text = text.replace('!', '')
    text = text.replace('名爵', 'mg')
    text = text.replace('上汽大通', '大通')
    text = text.replace('风行', '东风风行')
    text = text.replace('swm斯威', '斯威')
    if df['brand_name'] == '北汽制造':
        text = text.replace('北汽', '')
    if df['brand_name'] == '北汽绅宝':
        text = text.replace('绅宝', '北汽绅宝')
    text = text.replace('汽车', '')
    text = text.replace(' ', '')

    return text


def process_final_text(df):
    """
    删除没用的字符
    """
    text = df['model_name'].replace(df['brand_name'], '')
    # text = text.lower()
    return df['brand_name']+text


class Predict(object):

    def __init__(self):
        """
        加载各类匹配表和模型
        """
        self.brand_map = pd.read_csv(path + 'predict/model/brand_map.csv')
        self.car_autohome_all = pd.read_csv(path + '../tmp/train/car_autohome_all.csv')

    def base_clean(self):
        """
        基本清洗
        """
        # 加载原始训练数据
        self.car_autohome_model = self.car_autohome_all.loc[:, ['brand_slug', 'model_slug', 'brand_name', 'model_name']]
        self.car_autohome_model = self.car_autohome_model.drop_duplicates(['brand_slug', 'model_slug']).reset_index(drop=True)
        self.car_autohome_model['brand_name'] = self.car_autohome_model.apply(process_brand_name, axis=1)
        self.car_autohome_model['model_name'] = self.car_autohome_model.apply(process_model_name, axis=1)
        self.car_autohome_model['final_text'] = self.car_autohome_model.apply(process_final_text, axis=1)
        self.car_autohome_model = self.car_autohome_model.sort_values(by=['brand_name', 'model_name']).reset_index(drop=True)
        self.car_autohome_model.to_csv(path + '../tmp/train/man_car_autohome.csv', index=False)
        # self.test['text'] = self.test.apply(delete_str_useless, args=('title',), axis=1)
        return

    def create_test_final(self, data):
        """
        生成预测数据
        """
        # 加载映射器
        with open(path + 'predict/model/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        test_final = []

        for i in range(0, len(data)):
            temp = []
            x = data['text'][i]
            temp.extend(x)
            test_final.append(' '.join(temp))

        test_sequences = tokenizer.texts_to_sequences(test_final)
        pred = pad_sequences(test_sequences, maxlen=als.MAX_SEQUENCE_LENGTH)
        return pred

    def predict_brand(self):
        """
        预测品牌
        """
        pred = self.create_test_final(self.test)

        stack = Stacking()
        model = stack.define_brand_model_structure(als.MAX_NB_WORDS, als.EMBEDDING_DIM, als.MAX_SEQUENCE_LENGTH,
                                                  als.NUM_LSTM,
                                                  als.RATE_DROP_LSTM, als.RATE_DROP_DENSE, als.NUM_DENSE, als.ACT,
                                                  len(self.brand_map))
        model.load_weights(path + 'predict/model/brand_model_weights.hdf5')
        preds = model.predict(pred, batch_size=1024, verbose=als.VERBOSE)

        # 根据预测阈值区分可预测和不可预测
        final = []
        for i in range(0, len(preds)):
            if max(preds[i]) < als.THRESHOLD_BRAND:
                final.append(-1)
            else:
                final.append(list(preds[i]).index(max(list(preds[i]))))
        final = pd.DataFrame(pd.Series(final), columns=['brand_id'])
        final = final.merge(self.brand_map, how='left', on=['brand_id'])
        final = final.rename(columns={'brand_slug': 'predict_brand_slug', 'brand_name': 'predict_brand_name'})
        self.test['predict_brand_slug'] = final['predict_brand_slug']
        self.test.reset_index(inplace=True, drop=True)

    def predict_model(self):
        """
        预测车型
        """
        result = pd.DataFrame()
        brand_slugs = list(set(self.test.predict_brand_slug.values))
        if np.NAN in brand_slugs:
            brand_slugs.remove(np.NAN)
        brand_slugs = list(set(brand_slugs).intersection(set(self.brand_map.brand_slug.values)))

        for brand_slug in brand_slugs:
            brand_data = self.test.loc[(self.test['predict_brand_slug'] == brand_slug), :]
            brand_data.reset_index(inplace=True, drop=True)
            model_map = pd.read_csv(path + 'predict/model/brand/' + str(brand_slug) + '/model_map.csv')
            # 如果只有一个车型直接赋值
            if len(model_map) == 1:
                brand_data['predict_model_slug'] = list(model_map.model_slug.values)[0]
                result = result.append(brand_data, ignore_index=True)
                continue
            # 预测
            pred = self.create_test_final(brand_data)
            stack = Stacking()
            model = stack.define_brand_model_structure(als.MAX_NB_WORDS, als.EMBEDDING_DIM, als.MAX_SEQUENCE_LENGTH,
                                                       als.NUM_LSTM,
                                                       als.RATE_DROP_LSTM, als.RATE_DROP_DENSE, als.NUM_DENSE, als.ACT,
                                                       len(model_map))
            model.load_weights(path + 'predict/model/brand/' + str(brand_slug) + '/model_model_weights.hdf5')
            preds = model.predict(pred, batch_size=1024, verbose=als.VERBOSE)

            # 根据预测阈值区分可预测和不可预测
            final = []
            for i in range(0, len(preds)):
                if max(preds[i]) < als.THRESHOLD_MODEL:
                    final.append(-1)
                else:
                    final.append(list(preds[i]).index(max(list(preds[i]))))
            final = pd.DataFrame(pd.Series(final), columns=['model_id'])
            final = final.merge(model_map, how='left', on=['model_id'])
            final = final.rename(columns={'model_slug': 'predict_model_slug'})
            brand_data['predict_model_slug'] = final['predict_model_slug']
            result = result.append(brand_data, ignore_index=True)
        result = result.loc[:, ['id', 'predict_model_slug']]
        self.test = self.test.merge(result, how='left', on=['id'])
        self.test.reset_index(inplace=True, drop=True)

    def predict_detail(self):
        """
        预测款型
        """
        result = pd.DataFrame()
        brand_slugs = list(set(self.test.predict_brand_slug.values))
        if np.NAN in brand_slugs:
            brand_slugs.remove(np.NAN)
        brand_slugs = list(set(brand_slugs).intersection(set(self.brand_map.brand_slug.values)))

        for brand_slug in brand_slugs:
            brand_data = self.test.loc[(self.test['predict_brand_slug'] == brand_slug), :]
            brand_data.reset_index(inplace=True, drop=True)
            detail_map = pd.read_csv(path + 'predict/model/brand/' + str(brand_slug) + '/detail_map.csv')
            # 如果只有一个车型直接赋值
            if len(detail_map) == 1:
                brand_data['predict_model_detail_slug'] = list(detail_map.detail_slug.values)[0]
                result = result.append(brand_data, ignore_index=True)
                continue
            # 预测
            pred = self.create_test_final(brand_data)
            stack = Stacking()
            model = stack.define_brand_model_structure(als.MAX_NB_WORDS, als.EMBEDDING_DIM, als.MAX_SEQUENCE_LENGTH,
                                                       als.NUM_LSTM,
                                                       als.RATE_DROP_LSTM, als.RATE_DROP_DENSE, als.NUM_DENSE, als.ACT,
                                                       len(detail_map))
            model.load_weights(path + 'predict/model/brand/' + str(brand_slug) + '/detail_model_weights.hdf5')
            preds = model.predict(pred, batch_size=1024, verbose=als.VERBOSE)

            # 根据预测阈值区分可预测和不可预测
            final = []
            for i in range(0, len(preds)):
                if max(preds[i]) < als.THRESHOLD_DETAIL:
                    final.append(-1)
                else:
                    final.append(list(preds[i]).index(max(list(preds[i]))))
            final = pd.DataFrame(pd.Series(final), columns=['detail_id'])
            final = final.merge(detail_map, how='left', on=['detail_id'])
            final = final.rename(columns={'detail_slug': 'predict_detail_slug'})
            brand_data['predict_detail_slug'] = final['predict_detail_slug']
            result = result.append(brand_data, ignore_index=True)
        result = result.loc[:, ['id', 'predict_detail_slug']]
        self.test = self.test.merge(result, how='left', on=['id'])
        self.test.reset_index(inplace=True, drop=True)

    def execute(self):
        """
        执行品牌,车型,款型的预测
        """
        try:
            # 基本清洗
            self.base_clean()
            # # 预测品牌
            # self.predict_brand()
            # # 预测车型
            # self.predict_model()
            # # 预测款型
            # self.predict_detail()
            #
            # self.test.to_csv(path + '../tmp/train/car_detail_info.csv', index=False)

        except Exception:
            raise PredictError(traceback.format_exc())






