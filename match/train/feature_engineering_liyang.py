from match.train import *


def process_brand_name(df):
    """
    删除没用的字符
    """
    text = df['brand_name']
    text = text.replace('MINI(迷你)', 'MINI')
    text = text.replace('RUF(如虎)', '如虎')
    text = text.replace('Pagani', '帕加尼')
    text = text.replace('柯尼赛格', '科尼赛克')
    text = text.replace('布嘉迪', '布加迪')
    text = text.replace('幻速', '北汽幻速')
    text = text.replace('昌河', '北汽昌河')
    text = text.replace('汽车', '')
    text = text.replace(' ', '')
    text = text.lower()
    return text


def process_model_name(df):
    """
    删除没用的字符
    """
    text = df['model_name']
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
    text = text.lower()
    text = text.replace('ⅲ', 'iii')
    text = text.replace('pagani', '帕加尼')
    text = text.replace('柯尼赛格', '科尼赛克')
    text = text.replace('科尼赛格', '科尼赛克')
    text = text.replace('布嘉迪', '布加迪')
    text = text.replace('吉利康迪全球鹰', '全球鹰')
    text = text.replace('汽车', '')
    if df['brand_name'] == '长丰猎豹':
        text = text.replace('猎豹', '长丰猎豹')
    if df['brand_name'] == '北汽制造':
        text = text.replace('北汽', '')
    if df['brand_name'] == '北汽绅宝':
        text = text.replace('北京绅宝', '北汽绅宝')
    if df['brand_name'] == '北汽幻速':
        text = text.replace('幻速', '北汽幻速')
    if df['brand_name'] == '北汽昌河':
        text = text.replace('昌河', '北汽昌河')
    text = text.replace(' ', '')
    # text = text.lower()
    return text


def process_final_text(df):
    """
    删除没用的字符
    """
    text = df['model_name'].replace(df['brand_name'], '')
    # text = text.lower()
    return df['brand_name']+text


class FeatureEngineering(object):

    def __init__(self):
        # 加载各类相关表
        self.open_category = pd.read_csv(path + '../tmp/train/open_category.csv')
        self.liyang_brand = self.open_category.loc[(self.open_category['parent'].isnull()), ['id', 'name', 'slug']].rename(columns={'name': 'brand_name', 'slug': 'parent'}).reset_index(drop=True)
        self.liyang_model = self.open_category.loc[(self.open_category['parent'].notnull()), ['id', 'name', 'slug', 'parent']].rename(columns={'name': 'model_name',
                     'slug': 'model_slug', 'id': 'model_id'}).reset_index(drop=True)
        self.liyang_model = self.liyang_model.merge(self.liyang_brand, how='left', on=['parent']).rename(columns={'id': 'brand_id'})
        self.liyang_model = self.liyang_model.loc[:, ['brand_id', 'model_id', 'brand_name', 'model_name', 'model_slug']]

    def execute(self):
        """
        特征工程处理
        """
        try:
            pass
            # 基本清洗
            self.base_clean()
            # # 生成语料库
            # self.create_word2vec()
            # # 创建语料库映射器
            # self.create_tokenizer()
        except Exception:
            raise FeatureEngineeringError(traceback.format_exc())

    def base_clean(self):
        """
        数据常规预处理
        """
        try:
            self.liyang_model['brand_name'] = self.liyang_model.apply(process_brand_name, axis=1)
            self.liyang_model['model_name'] = self.liyang_model.apply(process_model_name, axis=1)
            self.liyang_model['final_text'] = self.liyang_model.apply(process_final_text, axis=1)
            self.liyang_model = self.liyang_model.sort_values(by=['brand_name', 'model_name']).reset_index(drop=True)
            self.liyang_model.to_csv(path + '../tmp/train/man_liyang.csv', index=False)

        except Exception:
            raise FeatureEngineeringError(traceback.format_exc())

    def create_word2vec(self):
        """
        生成语料库
        """
        try:
            texts = []

            for i in range(0, len(self.car_autohome_all)):
                texts.extend(self.car_autohome_all['final_text'][i])

            model = Word2Vec(texts, min_count=1, size=100)
            os.makedirs(os.path.dirname(path + 'predict/model/word2vec.bin'), exist_ok=True)
            model.save(path + 'predict/model/word2vec.bin')
        except Exception:
            raise FeatureEngineeringError(traceback.format_exc())

    def create_tokenizer(self):
        """
        创建语料库映射器
        """
        train_final = []

        for i in range(0, len(self.car_autohome_all)):
            temp = []
            x = self.car_autohome_all['final_text'][i]
            temp.extend(x)
            train_final.append(' '.join(temp))

        # 创建Tokenizer
        tokenizer = Tokenizer(num_words=als.MAX_NB_WORDS, lower=False)
        tokenizer.fit_on_texts(train_final)
        word_index = tokenizer.word_index
        print('Found %s unique tokens' % len(word_index))

        # 保存映射器和词典
        with open(path + 'predict/model/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        f = open(path + 'predict/model/word_index.txt', 'w')
        f.write(str(word_index))
        f.close()
