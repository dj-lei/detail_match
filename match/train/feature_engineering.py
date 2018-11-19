from match.train import *


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
    text = text.replace('\xa0', '')
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
    text = text.lower()
    return text


def final_process(df):
    """
    最终处理
    """
    text = df['detail_model']
    text = df['global_name']+text
    text = text.replace(' ', '')
    return text


def all_arrange(n, begin, end, result):
    """
    list全排列
    """
    if begin >= end:
        result.append(' '.join(n))
    else:
        i = begin
        for num in range(begin, end):
            n[num], n[i] = n[i], n[num]
            all_arrange(n, begin+1, end, result)
            n[num], n[i] = n[i], n[num]


class FeatureEngineering(object):

    def __init__(self):
        self.train = []
        # 加载各类相关表
        self.open_category = pd.read_csv(path + '../tmp/train/open_category.csv', encoding='gb18030')
        self.open_model_detail = pd.read_csv(path + '../tmp/train/open_model_detail.csv', encoding='gb18030')

    def execute(self):
        """
        特征工程处理
        """
        try:
            # 基本清洗
            self.base_clean()
            # 生成语料库
            self.create_word2vec()
        except Exception:
            raise FeatureEngineeringError(traceback.format_exc())

    def base_clean(self):
        """
        数据常规预处理
        """
        try:
            # 处理open_model_detail表text字段
            open_category = self.open_category.loc[(self.open_category['parent'].notnull()), :]
            open_category = open_category.rename(
                columns={'slug': 'global_slug', 'parent': 'brand_slug', 'name': 'global_name'})
            open_category = open_category.loc[:, ['global_slug', 'brand_slug', 'global_name']]

            open_model_detail = self.open_model_detail.merge(open_category, how='left', on=['global_slug'])
            open_model_detail = open_model_detail.loc[(open_model_detail['brand_slug'].notnull()) & (
                open_model_detail['global_slug'].notnull()), :]
            open_model_detail.reset_index(inplace=True, drop=True)
            open_model_detail['detail_model'] = open_model_detail.apply(delete_str_useless, args=('detail_model',),
                                                                        axis=1)
            open_model_detail['global_name'] = open_model_detail.apply(delete_str_useless, args=('global_name',),
                                                                       axis=1)

            final = pd.DataFrame([], columns=['detail_model', 'model_detail_slug'])
            final.to_csv('../tmp/train/step1.csv', index=False)
            for i in range(0, len(open_model_detail)):
                detail_model = open_model_detail.loc[i, 'detail_model']
                detail_model = detail_model.split(' ')
                result = []
                all_arrange(detail_model, 0, len(detail_model), result)
                temp = pd.DataFrame(pd.Series(result), columns=['detail_model'])
                temp['model_detail_slug'] = open_model_detail.loc[i, 'model_detail_slug']
                final = final.append(temp)
                final.reset_index(inplace=True, drop=True)
                if (i % 1000) == 0:
                    final.to_csv('../tmp/train/step1.csv', header=False, mode='a', index=False)
                    final = pd.DataFrame([], columns=['detail_model', 'model_detail_slug'])
                elif i == (len(open_model_detail) - 1):
                    final.to_csv('../tmp/train/step1.csv', header=False, mode='a', index=False)

            final = pd.read_csv('../tmp/train/step1.csv', encoding='gb18030')
            open_model_detail = open_model_detail.drop(['detail_model'], axis=1)
            self.train = open_model_detail.merge(final, how='left', on=['model_detail_slug'])
            self.train.reset_index(inplace=True, drop=True)
            self.train['final_text'] = self.train.apply(final_process, axis=1)
            self.train = self.train.loc[:, ['brand_slug', 'global_slug', 'model_detail_slug', 'year', 'final_text']]
            # 存储中间文件
            self.train.to_csv(path + '../tmp/train/train_final.csv', index=False)

        except Exception:
            raise FeatureEngineeringError(traceback.format_exc())

    def create_word2vec(self):
        """
        生成语料库
        """
        try:
            texts = []

            for i in range(0, len(self.train)):
                texts.extend(self.train['final_text'][i])

            model = Word2Vec(texts, min_count=1, size=100)
            os.makedirs(os.path.dirname(path + 'predict/model/word2vec.bin'), exist_ok=True)
            model.save(path + 'predict/model/word2vec.bin')
        except Exception:
            raise FeatureEngineeringError(traceback.format_exc())

