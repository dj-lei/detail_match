from match.train import *


def delete_str_useless(text):
    """
    删除没用的字符
    """
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
    text = text.replace(' ', '')
    return text


def clean_normal_str(df):
    """
    清洗model_detail_normal表款型详情
    """
    text = df['model_detail']
    model = df['model']
    model = model.lower()
    text = text.lower()
    text = re.sub(r"\d+款", '', text)
    text = text.replace(model, '')
    text1 = str(df['year'])+'款'+str(df['global_name']).lower()+text
    text2 = str(df['year'])+'款'+model+text
    text1 = delete_str_useless(text1)
    text2 = delete_str_useless(text2)
    return pd.Series([text1, text2])


def clean_detail_str(df):
    """
    清洗open_model_detail表款型详情
    """
    text = df['detail_model']
    text = text.lower()
    text = re.sub(r"\d+款", '', text)
    text = str(df['year'])+'款'+str(df['global_name']).lower()+text
    text = delete_str_useless(text)
    return text


class FeatureEngineering(object):

    def __init__(self):
        self.train = []
        # 加载各类相关表
        self.open_model_detail = pd.read_csv(path+'../tmp/train/open_model_detail.csv')
        self.open_model_detail = self.open_model_detail[self.open_model_detail['price_bn'] > 0]
        self.open_model_detail.reset_index(inplace=True, drop=True)
        self.open_category = pd.read_csv(path+'../tmp/train/open_category.csv')
        self.normal = pd.read_csv(path+'../tmp/train/model_detail_normal.csv')
        # self.domain_priority = pd.read_csv(path + '../tmp/train/domain_priority.csv')

    def execute(self):
        """
        特征工程处理
        """
        try:
            # 基本清洗
            self.base_clean()
            # 生成语料库
            self.create_word2vec()
        except Exception as e:
            raise FeatureEngineeringError(traceback.format_exc())

    def base_clean(self):
        """
        数据常规预处理
        """
        try:
            # 处理model_detail_normal表text字段
            open_category = self.open_category.loc[(self.open_category['parent'].notnull()), :]
            open_category = open_category.rename(columns={'slug': 'global_slug', 'parent': 'brand_slug'})
            open_category = open_category.loc[:, ['global_slug', 'brand_slug']]
            open_model_detail = self.open_model_detail.loc[:, ['id', 'model_detail_slug', 'global_slug']]
            open_model_detail = open_model_detail.rename(columns={'id': 'model_detail_slug_id'})
            open_model_detail = open_model_detail.merge(open_category, how='left', on=['global_slug'])

            normal = self.normal.loc[(self.normal['model_detail_slug_id'].notnull()), :]
            normal['model_detail_slug_id'] = normal['model_detail_slug_id'].astype(int)
            normal = normal.drop(['global_slug'], axis=1)
            normal = normal.merge(open_model_detail, how='left', on=['model_detail_slug_id'])
            normal = normal.loc[(normal['brand_slug'].notnull()) & (normal['global_slug'].notnull()) & (normal['model_detail_slug_id'].notnull()), :]
            normal.reset_index(inplace=True, drop=True)
            normal[['final_text', 'final_text_version2']] = normal.apply(clean_normal_str, axis=1)
            normal = normal.loc[(normal['model_detail_slug'].notnull()), :]
            normal1 = normal.loc[:, ['brand_slug', 'global_slug', 'model_detail_slug', 'year', 'final_text']]
            normal2 = normal.loc[:, ['brand_slug', 'global_slug', 'model_detail_slug', 'year', 'final_text_version2']]
            normal2 = normal2.rename(columns={'final_text_version2': 'final_text'})

            # 处理open_model_detail表text字段
            open_category = self.open_category.loc[(self.open_category['parent'].notnull()), :]
            open_category = open_category.rename(columns={'slug': 'global_slug', 'parent': 'brand_slug', 'name': 'global_name'})
            open_category = open_category.loc[:, ['global_slug', 'brand_slug', 'global_name']]

            open_model_detail = self.open_model_detail.merge(open_category, how='left', on=['global_slug'])
            open_model_detail = open_model_detail.loc[(open_model_detail['brand_slug'].notnull()) & (open_model_detail['global_slug'].notnull()), :]
            open_model_detail.reset_index(inplace=True, drop=True)
            open_model_detail['final_text'] = open_model_detail.apply(clean_detail_str, axis=1)
            open_model_detail = open_model_detail.loc[:, ['brand_slug', 'global_slug', 'model_detail_slug', 'year', 'final_text']]

            # 存储中间文件
            self.train = normal1.append(normal2)
            self.train = self.train.append(open_model_detail)
            self.train.reset_index(inplace=True, drop=True)
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