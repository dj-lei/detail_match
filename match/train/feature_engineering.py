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
    text = df['detail_name']
    text = df['model_name']+text
    text = text.replace(' ', '')
    return text


def cal_online_year(df):
    """
    查找年款
    """
    regex = re.compile("(\d+)款")
    result = regex.findall(df['detail_name'])
    return result[0]


class FeatureEngineering(object):

    def __init__(self):
        # 加载各类相关表
        self.car_autohome_all = pd.read_csv(path + '../tmp/train/car_autohome_all.csv')

    def execute(self):
        """
        特征工程处理
        """
        try:
            # 基本清洗
            self.base_clean()
            # 生成语料库
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
            self.car_autohome_all['detail_name'] = self.car_autohome_all.apply(delete_str_useless, args=('detail_name', ), axis=1)
            self.car_autohome_all['online_year'] = self.car_autohome_all.apply(cal_online_year, axis=1)
            self.car_autohome_all['final_text'] = self.car_autohome_all.apply(final_process, axis=1)
            self.car_autohome_all = self.car_autohome_all.loc[:, ['brand_slug', 'brand_name', 'model_slug', 'model_name', 'detail_slug', 'detail_name', 'online_year', 'final_text']]
            # 存储中间文件
            self.car_autohome_all.to_csv(path + '../tmp/train/train_final.csv', index=False)

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
