from match.match import *


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


def find_brand(df, brand):
    for i in range(0, len(brand)):
        if brand.loc[i, 'brand_name'] in gl.ENGLISH_BRAND:
            rule = '^' + brand.loc[i, 'brand_name']
            match = re.findall(rule, df['detail_name'])
            if len(match) > 0:
                return pd.Series([brand.loc[(brand['brand_name'] == match[0]),'brand_name'].values[0], brand.loc[(brand['brand_name'] == match[0]),'brand_slug'].values[0]])
        else:
            match = re.findall(brand.loc[i,'brand_name'],df['detail_name'])
            if len(match) > 0:
                return pd.Series([brand.loc[i,'brand_name'],brand.loc[i,'brand_slug']])
    raise ApiParamsError('款型描述需包含品牌+车系+年款和必要描述(尽量包含变速器和排量)!例如:吉利 远景suv 2016款 1.3t 无级 旗舰')


def process_online_year(sentence):
    temp = re.findall('(\d\d\d\d)款', sentence)
    if len(temp) == 0:
        raise ApiParamsError('款型描述必须包含年款!例如:吉利 远景suv 2016款 1.3t 无级 旗舰')
    return temp[0]


def process_volume(sentence):
    temp = re.findall('(\d\.\d)', sentence)
    if len(temp) == 0:
        return np.NaN
    return temp[0]


def process_control(sentence):
    text = sentence
    text = text.lower()
    temp = re.findall('自动|手自|amt|无级|cvt|双离合|dct|dsg', text)
    if len(temp) > 0:
        return '自动'
    temp = re.findall('手动', text)
    if len(temp) > 0:
        return '手动'
    return np.NaN


def cal_cos_similar(df, y_vector):
    x_vector = eval(df['fill_vector'])
    y_vector = eval(y_vector)
    cos_similar = np.dot(x_vector, y_vector) / (np.linalg.norm(x_vector, ord=2) * np.linalg.norm(y_vector, ord=2))
    return float('%.3f' % cos_similar)


def match_detail(df, car_autohome_all, columns):
    # print(df['brand_slug'], df['online_year'], df['control'], df['volume'])
    brand_slug = eval(df['brand_slug'])
    if (type(df['control']) != str) & math.isnan(float(df['volume'])):
        temp_detail = car_autohome_all.loc[(car_autohome_all['brand_slug'].isin(brand_slug)) & (
                    car_autohome_all['online_year'] == int(df['online_year'])), :].reset_index(drop=True)
    elif (type(df['control']) != str) & (not math.isnan(float(df['volume']))):
        temp_detail = car_autohome_all.loc[(car_autohome_all['brand_slug'].isin(brand_slug)) & (
                    car_autohome_all['online_year'] == int(df['online_year'])) & (
                                                       car_autohome_all['volume'] == float(df['volume'])),
                      :].reset_index(drop=True)
    elif (type(df['control']) == str) & (math.isnan(float(df['volume']))):
        temp_detail = car_autohome_all.loc[(car_autohome_all['brand_slug'].isin(brand_slug)) & (
                    car_autohome_all['online_year'] == int(df['online_year'])) & (
                                                       car_autohome_all['control'] == df['control']), :].reset_index(
            drop=True)
    else:
        temp_detail = car_autohome_all.loc[(car_autohome_all['brand_slug'].isin(brand_slug)) & (
                    car_autohome_all['online_year'] == int(df['online_year'])) & (
                                                       car_autohome_all['control'] == df['control']) & (
                                                       car_autohome_all['volume'] == float(df['volume'])),
                      :].reset_index(drop=True)

    if len(temp_detail) == 0:
        print('no', df['brand_name'], df['brand_slug'], df['online_year'], df['control'], df['volume'])
        raise ApiParamsError('款型描述需包含品牌+车系+年款和必要描述(尽量包含变速器和排量)!例如:吉利 远景suv 2016款 1.3t 无级 旗舰')
    temp_detail['cos_similar'] = temp_detail.apply(cal_cos_similar, args=(df['fill_vector'],), axis=1)
    temp_detail = temp_detail.loc[temp_detail['cos_similar'] == max(list(set(temp_detail.cos_similar.values))), :]
    result = temp_detail.loc[temp_detail.groupby(['brand_slug']).cos_similar.idxmax(), columns].values[0]
    return pd.Series(result)


def texts_to_sequences(sentence, word_index):
    """
    词向量转序列
    """
    vector = []
    word_index_copy = word_index.copy()
    word_index_copy.index = word_index_copy['word']
    for char in sentence:
        try:
            vector.append(int(word_index_copy['num'][char]))
        except Exception:
            continue
    return vector


class Match(object):

    def __init__(self):
        self.word_index = pd.read_csv(path + '../tmp/train/word_index.csv')
        self.brand = pd.read_csv(path + '../tmp/train/brand_name.csv')
        self.car_autohome_cos_vector = pd.read_csv(path + '../tmp/train/car_autohome_cos_vector.csv')
        self.columns = list(self.car_autohome_cos_vector.columns)
        self.columns.extend(['cos_similar'])

    def predict(self, detail_name='吉利 远景suv 2016款 1.3t 无级 旗舰', cos_similar=0.82):
        """
        预测款型
        """
        result = {'status': 'fail', 'message': '', 'data': ''}
        try:
            # 款型匹配
            params = pd.DataFrame([[detail_name]])
            params.columns = ['detail_name']

            params['online_year'] = process_online_year(params['detail_name'][0])
            params['volume'] = process_volume(params['detail_name'][0])
            params['control'] = process_control(params['detail_name'][0])

            params['origin_name'] = params['detail_name']
            params['detail_name'] = params.apply(delete_str_useless, args=('detail_name',), axis=1)
            params[['brand_name', 'brand_slug']] = params.apply(find_brand, args=(self.brand, ), axis=1)

            test_final = params['detail_name'][0]
            test_sequences = texts_to_sequences(test_final, self.word_index)
            text = [0 for i in range(0, len(self.word_index))]

            for j in test_sequences:
                text[j - 1] = text[j - 1] + 1
            params.loc[0, 'fill_vector'] = str(text)
            params[self.columns] = params.apply(match_detail, args=(self.car_autohome_cos_vector, self.columns, ), axis=1)

            if params['cos_similar'][0] < cos_similar:
                result['message'] = '匹配相似度低于阈值,已过滤!'
                return result

            params = params.drop(['fill_vector', 'listed_year', 'manufacturer', 'body_style', 'emission_standard', 'popular'], axis=1)
            result['status'] = 'success'
            result['data'] = params.to_dict('records')[0]
            return result
        except Exception as e:
            result['message'] = e.message
            return result





