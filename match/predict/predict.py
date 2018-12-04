from match.predict import *


def delete_str_useless(df, column_name):
    """
    删除没用的字符
    """
    text = df[column_name]
    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace('【', '')
    text = text.replace('】', '')
    text = text.replace('（', '')
    text = text.replace('）', '')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('_', '')
    text = text.replace('-', '')
    text = text.replace('+', '')
    text = text.replace('—', '')
    text = text.replace('/', '')
    text = text.replace('“', '')
    text = text.replace('”', '')
    text = text.replace('!', '')
    text = text.replace('。', '')
    text = text.replace('＞', '')
    text = text.replace('・', '')
    text = text.replace('》', '')
    text = text.replace('！', '')
    text = text.replace('／', '')
    text = text.replace('’', '')
    text = text.replace('－', '')
    text = text.replace('•', '')
    text = text.replace('×', '')
    text = text.replace('《', '')
    text = text.replace('＿', '')
    text = text.replace('®', '')
    text = text.replace('─', '')
    text = text.replace('、', '')
    text = text.replace(' ', '')
    text = text.lower()
    return text


class Predict(object):

    def __init__(self):
        """
        加载各类匹配表和模型
        """
        self.test = pd.DataFrame()
        self.brand_map = pd.read_csv(path + 'predict/model/brand_map.csv')
        # self.exception_model = pd.read_csv(path + 'predict/model/brand/exception_model_predict.csv')
        # self.exception_detail = pd.read_csv(path + 'predict/model/brand/exception_detail_predict.csv')

    def base_clean(self):
        """
        基本清洗
        """
        # 加载原始训练数据
        self.test = pd.read_csv(path + '../tmp/train/predict.csv')
        # self.test = self.test.loc[(~self.test['brand_slug'].isin(list(self.exception_model.exception_brand.values))), :]
        # self.test = self.test.loc[(~self.test['brand_slug'].isin(list(self.exception_detail.exception_brand.values))), :]
        # self.test.reset_index(inplace=True, drop=True)
        self.test['text'] = self.test.apply(delete_str_useless, args=('title',), axis=1)
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
            # 预测品牌
            self.predict_brand()
            # 预测车型
            self.predict_model()
            # 预测款型
            self.predict_detail()

            # self.test = self.test.drop(['id', 'price_bn', 'year', 'volume', 'control', 'detail_model','brand_slug'], axis=1)
            self.test.to_csv(path + '../tmp/train/car_detail_info.csv', index=False)

        except Exception:
            raise PredictError(traceback.format_exc())






