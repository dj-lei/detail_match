from match.generate import *


def delete_str_useless(df, column_name):
    """
    删除没用的字符
    """
    text = df[column_name]
    text = text.lower()
    text = text.replace('ⅰ', 'i')
    text = text.replace('ⅱ', 'ii')
    text = text.replace('ⅲ', 'iii')
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
    # text = text.replace('+', '')
    text = text.replace('—', '')
    text = text.replace('/', '')
    text = text.replace('“', '')
    text = text.replace('”', '')
    text = text.replace('!', '')
    text = text.replace('。', '')
    text = text.replace('＞', '')
    text = text.replace('·', '')
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


def cal_brand_count(df):
    return len(df['brand_name'])


def add_brand(brand):
    brand = brand.loc[~(brand['brand_name'].isin(['长安', '长安跨越', '长安欧尚', '长安轻型车', '北京', '迈巴赫', '福田', '福田乘用车', '三菱', '东风小康'])), :].reset_index(drop=True)
    brand = brand.append(pd.DataFrame([['昌河',79],['道达',301],['幻速',203],['绅宝',173],['威旺',143],['吉利',25],['swm斯威',269],['斯威',269],['猎豹',78],
                                      ['力帆',80],['五菱',114],['野马',111],['mg',20],['英致',192],['汉腾',267],['君马',297],['大通',155],['比速', 271],
                                      ['吉奥',108],['康迪',219],['夏利',110],['莲花',89],['迈巴赫',55],['吉普',46],['东风小康', 142]],columns=['brand_name','brand_slug']),sort=False).reset_index(drop=True)
    return brand


def process_brand_slug(df):
    return str([df['brand_slug']])


def fill_dup_brand_slug(brand):
    brand = brand.append(pd.DataFrame([['传祺', '[82, 313]'], ['三菱', '[68,329]'], ['福田', '[96,282]'],['长安', '[294,299,163,76]'],['北京', '[27,154]'],['北汽', '[27,154]']],columns=['brand_name','brand_slug']),sort=False).reset_index(drop=True)
    return brand


class Generate(object):

    def __init__(self):
        # 加载各类相关表
        self.car_autohome_all = pd.read_csv(path + '../tmp/train/car_autohome_all.csv')
        self.gpj_model = pd.read_csv(path + '../tmp/train/combine_model.csv', low_memory=False)
        self.gpj_detail = pd.read_csv(path + '../tmp/train/combine_detail.csv', low_memory=False)

    def generate_word_vector_map(self):
        """
        生成词向量映射
        """
        car_autohome_all = self.car_autohome_all.copy()
        car_autohome_all['final_text'] = car_autohome_all['brand_name'] + car_autohome_all['model_name'] + car_autohome_all['detail_name']
        car_autohome_all['final_text'] = car_autohome_all.apply(delete_str_useless, args=('final_text',), axis=1)

        texts = []
        train_final = []
        generate_text_index = car_autohome_all.copy()
        for i in range(0, len(generate_text_index)):
            temp = []
            texts.extend(generate_text_index['final_text'][i])
            x = generate_text_index['final_text'][i]
            temp.extend(x)
            train_final.append(temp)

        # 创建Tokenizer
        tokenizer = Tokenizer(num_words=len(set(texts)), char_level=True, lower=True)
        tokenizer.fit_on_texts(train_final)
        word_index = tokenizer.word_index
        print('Found %s unique tokens' % len(word_index))

        # 保存映射器和词典
        with open(path + '../tmp/train/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        f = open(path + '../tmp/train/word_index.txt', 'w')
        f.write(str(word_index))
        f.close()

    def generate_standard_cos_vector(self):
        """
        生成标准库余弦向量
        """
        f = open(path + '../tmp/train/word_index.txt', 'r', encoding='UTF-8')
        temp = f.read()
        word_index = eval(temp)

        with open(path + '../tmp/train/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        car_autohome_all = self.car_autohome_all.copy()
        car_autohome_all['origin_brand_name'] = car_autohome_all['brand_name']
        car_autohome_all['origin_model_name'] = car_autohome_all['model_name']
        car_autohome_all['brand_name'] = car_autohome_all.apply(delete_str_useless, args=('brand_name',), axis=1)
        car_autohome_all['model_name'] = car_autohome_all.apply(delete_str_useless, args=('model_name',), axis=1)

        # part1
        part1 = car_autohome_all.copy()
        part1['final_text'] = part1['brand_name'] + part1['model_name'] + part1['detail_name']
        part1['final_text'] = part1.apply(delete_str_useless, args=('final_text',), axis=1)

        # part2
        part2 = car_autohome_all.copy()
        part2['final_text'] = part2['model_name'] + part2['detail_name']
        part2['final_text'] = part2.apply(delete_str_useless, args=('final_text',), axis=1)

        car_autohome_all = part1.append(part2, sort=False).reset_index(drop=True)
        car_autohome_all['fill_vector'] = np.NaN

        train_final = []
        for i in range(0, len(car_autohome_all)):
            temp = []
            x = car_autohome_all['final_text'][i]
            temp.extend(x)
            train_final.append(temp)
        train_sequences = tokenizer.texts_to_sequences(train_final)

        fill_zero = [0 for i in range(0, len(word_index))]

        for i in range(0, len(car_autohome_all)):
            text = fill_zero.copy()
            for j in train_sequences[i]:
                text[j - 1] = text[j - 1] + 1
            car_autohome_all.loc[i, 'fill_vector'] = str(text)
        car_autohome_all['brand_name'] = car_autohome_all['origin_brand_name']
        car_autohome_all['model_name'] = car_autohome_all['origin_model_name']
        car_autohome_all = car_autohome_all.drop(['origin_brand_name', 'origin_model_name', 'volume_extend', 'final_text'], axis=1)
        car_autohome_all.to_csv(path + '../tmp/train/car_autohome_cos_vector.csv', index=False)

    def match_gpj_detail(self):
        """
        匹配公平价款型
        """
        car_autohome_all = pd.read_csv(path + '../tmp/train/car_autohome_cos_vector.csv')
        gpj_model = self.gpj_model.copy()
        gpj_model = gpj_model.loc[:, ['slug', 'parent', 'classified_slug', 'popular']]
        gpj_model = gpj_model.rename(columns={'slug': 'gpj_model_slug', 'parent': 'gpj_brand_slug', 'classified_slug': 'body_style'})

        gpj_detail = self.gpj_detail.copy()
        gpj_detail = gpj_detail.loc[:, ['global_slug', 'car_autohome_detail_id', 'detail_model_slug']]
        gpj_detail = gpj_detail.rename(columns={'global_slug': 'gpj_model_slug', 'car_autohome_detail_id': 'detail_slug', 'detail_model_slug': 'gpj_detail_slug'})
        gpj_detail = gpj_detail.merge(gpj_model, how='left', on=['gpj_model_slug'])

        car_autohome_all = car_autohome_all.merge(gpj_detail, how='left', on=['detail_slug'])
        car_autohome_all.to_csv(path + '../tmp/train/car_autohome_cos_vector.csv', index=False)

    def update_tables(self):
        """
        更新相关表
        """
        car_autohome_cos_vector = pd.read_csv(path + '../tmp/train/car_autohome_cos_vector.csv')
        process_tables.insert_or_update_match_cos_vector(car_autohome_cos_vector)

        brand = car_autohome_cos_vector.loc[:, ['brand_name', 'brand_slug']].drop_duplicates(['brand_name', 'brand_slug']).reset_index(drop=True)
        brand['brand_name'] = brand.apply(delete_str_useless, args=('brand_name',), axis=1)
        brand['name_count'] = brand.apply(cal_brand_count, axis=1)
        brand = brand.sort_values(by=['name_count'], ascending=False).reset_index(drop=True)
        brand = brand.loc[:, ['brand_name', 'brand_slug']].drop_duplicates(['brand_name']).reset_index(drop=True)
        brand = add_brand(brand)
        brand['brand_slug'] = brand.apply(process_brand_slug, axis=1)
        brand = fill_dup_brand_slug(brand)
        brand.to_csv(path + '../tmp/train/brand_name.csv', index=False)
        process_tables.insert_or_update_match_brand_name(brand)

        f = open(path + '../tmp/train/word_index.txt', 'r', encoding='UTF-8')
        temp = f.read()
        word_index = eval(temp)
        word_index = pd.DataFrame.from_dict(word_index, orient='index').reset_index()
        word_index.columns = ['word', 'num']
        word_index.to_csv(path + '../tmp/train/word_index.csv', index=False)
        process_tables.insert_or_update_match_word_index(word_index)

    def execute(self):
        self.generate_word_vector_map()
        self.generate_standard_cos_vector()
        self.match_gpj_detail()
        self.update_tables()