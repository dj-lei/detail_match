from match.train import *


def delete_str_useless(df, column_name):
    """
    删除没用的字符
    """
    text = df[column_name]
    # text = text.replace('[', '')
    # text = text.replace(']', '')
    # text = text.replace('【', '')
    # text = text.replace('】', '')
    # text = text.replace('（', '')
    # text = text.replace('）', '')
    # text = text.replace('(', '')
    # text = text.replace(')', '')
    # text = text.replace('_', '')
    text = text.replace('-', '')
    # text = text.replace('+', '')
    # text = text.replace('—', '')
    # text = text.replace('/', '')
    # text = text.replace('“', '')
    # text = text.replace('”', '')
    # text = text.replace('!', '')
    # text = text.replace('。', '')
    # text = text.replace('＞', '')
    text = text.replace('・', '')
    # text = text.replace('》', '')
    # text = text.replace('！', '')
    # text = text.replace('／', '')
    # text = text.replace('’', '')
    # text = text.replace('－', '')
    # text = text.replace('•', '')
    # text = text.replace('×', '')
    # text = text.replace('《', '')
    # text = text.replace('＿', '')
    # text = text.replace('®', '')
    # text = text.replace('─', '')
    # text = text.replace('、', '')
    text = text.replace(' ', '')
    text = text.lower()
    return text


def final_process(df):
    """
    最终处理
    """
    text = re.sub("汽车", '', df['brand_name']) + ' ' + re.sub(r"一汽-", '', df['model_name']) + ' ' + df['detail_name']
    return text


def cal_online_year(df):
    """
    查找年款
    """
    regex = re.compile("(\d+)款")
    result = regex.findall(df['detail_name'])
    return result[0]


def replace_jinkou_position(df):
    """
    替换进口位置
    """
    return re.sub("\(进口\)", '', df['final_text']) + '(进口)'


def replace_english_char_position(df):
    """
    替换进口位置
    """
    text = re.findall(r'[\u4E00-\u9FA5]+$', df['model_name'])[0]
    return df['brand_name'] + ' ' + text + ' ' + df['detail_name']


def replace_brand_contain_model(df):
    """
    品牌包含在车系
    """
    text = re.sub("\(进口\)", '', df['model_name'])
    text = re.sub(r"一汽-", '', text)
    return text + ' ' + df['detail_name']


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
            self.car_autohome_all['online_year'] = self.car_autohome_all.apply(cal_online_year, axis=1)
            self.car_autohome_all['final_text'] = self.car_autohome_all.apply(final_process, axis=1)

            supplement_part3 = self.car_autohome_all.copy()
            # 品牌包含在车系名称
            supplement_part3['final_text'] = supplement_part3.apply(replace_brand_contain_model, axis=1)
            self.car_autohome_all = self.car_autohome_all.append(supplement_part3, sort=False).reset_index(drop=True)

            supplement_part2 = pd.DataFrame()
            # 补充车型名称变化
            regex = re.compile(r'[a-zA-Z].*[\u4E00-\u9FA5]$')
            for i in range(0, len(self.car_autohome_all)):
                text = regex.findall(self.car_autohome_all.loc[i, 'model_name'])
                if len(text) != 0:
                    supplement_part2 = supplement_part2.append(self.car_autohome_all.loc[i, :], sort=False)
            supplement_part2.reset_index(inplace=True, drop=True)
            supplement_part2['final_text'] = supplement_part2.apply(replace_english_char_position, axis=1)
            self.car_autohome_all = self.car_autohome_all.append(supplement_part2, sort=False).reset_index(drop=True)

            supplement_part1 = pd.DataFrame()
            # 将进口放在末尾
            regex = re.compile("\(进口\)")
            for i in range(0, len(self.car_autohome_all)):
                if (len(regex.findall(self.car_autohome_all.loc[i, 'manufacturer'])) != 0) | (len(regex.findall(self.car_autohome_all.loc[i, 'model_name'])) != 0):
                    supplement_part1 = supplement_part1.append(self.car_autohome_all.loc[i, :], sort=False)
            supplement_part1.reset_index(inplace=True, drop=True)
            supplement_part1['final_text'] = supplement_part1.apply(replace_jinkou_position, axis=1)
            self.car_autohome_all = self.car_autohome_all.append(supplement_part1, sort=False).reset_index(drop=True)

            self.car_autohome_all['final_text'] = self.car_autohome_all.apply(delete_str_useless, args=('final_text',), axis=1)
            self.car_autohome_all = self.car_autohome_all.drop_duplicates(['final_text'])
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
