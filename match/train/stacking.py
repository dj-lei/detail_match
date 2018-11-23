from match.train import *


def cal_train_param(data):
    """
    计算训练参数
    """
    batch_size = int(len(data) / 140)
    if batch_size < 8:
        batch_size = 8
    elif batch_size > 1024:
        batch_size = 1024
    return batch_size


class Stacking(object):
    """
    组合模型训练
    """
    def __init__(self):
        self.X = []
        self.Y = []
        self.train_final = []
        self.labels_final = []
        self.word_index = []
        self.train = []
        self.word2vec = Word2Vec.load(path + 'predict/model/word2vec.bin')
        self.open_category = []
        self.brand = []
        self.model = []
        self.open_model_detail = []
        self.details = []

    def init_variable(self):
        """
        初始化变量
        """
        self.train = pd.read_csv(path + '../tmp/train/train_final.csv')
        self.word2vec = Word2Vec.load(path + 'predict/model/word2vec.bin')
        self.open_category = pd.read_csv(path + '../tmp/train/open_category.csv')
        self.brand = self.open_category.loc[(self.open_category['parent'].isnull()), ['name', 'slug']].rename(
            columns={'name': 'brand_name', 'slug': 'brand_slug'})
        self.model = self.open_category.loc[
            (self.open_category['parent'].notnull()), ['name', 'slug', 'parent']].rename(
            columns={'name': 'global_name', 'slug': 'global_slug', 'parent': 'brand_slug'})
        self.model = self.model.merge(self.brand, how='left', on=['brand_slug'])
        self.open_model_detail = pd.read_csv(path + '../tmp/train/open_model_detail.csv')
        self.details = self.open_model_detail.loc[:, ['detail_model', 'model_detail_slug', 'global_slug']].rename(
            columns={'detail_model': 'detail_name'})
        self.details = self.details.merge(self.model, how='left', on=['global_slug'])

    def define_predict_map(self):
        """
        定义预测匹配表
        """
        brand_map = pd.DataFrame(pd.Series(list(self.brand.brand_slug.values)), columns=['brand_slug']).reset_index()
        brand_map = brand_map.rename(columns={'index': 'brand_id'})
        model_map = pd.DataFrame(pd.Series(list(self.model.global_slug.values)), columns=['global_slug']).reset_index()
        model_map = model_map.rename(columns={'index': 'model_id'})
        detail_map = pd.DataFrame(pd.Series(list(self.details.model_detail_slug.values)), columns=['model_detail_slug']).reset_index()
        detail_map = detail_map.rename(columns={'index': 'detail_id'})

        os.makedirs(os.path.dirname(path + 'predict/map/brand_map.csv'), exist_ok=True)
        brand_map = brand_map.merge(self.brand, how='left', on=['brand_slug'])
        brand_map = brand_map.loc[(brand_map['brand_slug'].isin(list(set(self.train.brand_slug.values)))), :]
        brand_map.reset_index(inplace=True, drop=True)
        brand_map = brand_map.drop(['brand_id'], axis=1).reset_index()
        brand_map = brand_map.rename(columns={'index': 'brand_id'})
        brand_map.to_csv(path + 'predict/map/brand_map.csv', index=False)
        model_map = model_map.merge(self.model, how='left', on=['global_slug'])
        model_map = model_map.loc[(model_map['global_slug'].isin(list(set(self.train.global_slug.values)))), :]
        model_map.reset_index(inplace=True, drop=True)
        model_map = model_map.drop(['model_id'], axis=1).reset_index()
        model_map = model_map.rename(columns={'index': 'model_id'})
        model_map.to_csv(path + 'predict/map/model_map.csv', index=False)
        detail_map = detail_map.merge(self.details, how='left', on=['model_detail_slug'])
        detail_map = detail_map.loc[(detail_map['model_detail_slug'].isin(list(set(self.train.model_detail_slug.values)))), :]
        detail_map.reset_index(inplace=True, drop=True)
        detail_map = detail_map.drop(['detail_id'], axis=1).reset_index()
        detail_map = detail_map.rename(columns={'index': 'detail_id'})
        detail_map.to_csv(path + 'predict/map/detail_map.csv', index=False)

        self.train = self.train.merge(brand_map, how='left', on=['brand_slug'])

    def combine_train_label(self, data, label_name):
        """
        组装训练数据和标签
        """
        self.train_final = []
        self.labels_final = []

        # 拷贝副本
        temp = data.copy()
        for i in range(0, als.COPY_DOSE):
            data = data.append(temp)
        data.reset_index(inplace=True, drop=True)

        for i in range(0, len(data)):
            temp = []
            x = data['final_text'][i]
            temp.extend(x)
            self.train_final.append(' '.join(temp))
            self.labels_final.append(data[label_name][i])

    def create_tokenizer(self):
        """
        创建语料库映射器
        """
        # 创建Tokenizer
        tokenizer = Tokenizer(num_words=als.MAX_NB_WORDS, lower=False)
        tokenizer.fit_on_texts(self.train_final)
        self.word_index = tokenizer.word_index
        print('Found %s unique tokens' % len(self.word_index))

        # 保存映射器和词典
        with open(path + 'predict/model/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        f = open(path + 'predict/model/word_index.txt', 'w')
        f.write(str(self.word_index))
        f.close()

    def define_brand_model_structure(self, max_nb_words, embedding_dim, max_sequence_length, num_lstm,
                                     rate_drop_lstm, rate_drop_dense, num_dense, act, category_num):
        """
        定义模型结构
        """
        # 加载词典
        f = open(path + 'predict/model/word_index.txt', 'r', encoding='UTF-8')
        # f = open(path + 'predict/model/word_index.txt', 'r', encoding='gb18030')
        temp = f.read()
        self.word_index = eval(temp)
        f.close()

        # 设置随机种子
        np.random.seed(als.SEED)

        embedding_matrix = np.zeros(((min(max_nb_words, len(self.word_index))+1), embedding_dim))
        for word, i in self.word_index.items():
            embedding_matrix[i] = self.word2vec[word]
        # print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

        embedding_layer = Embedding((min(max_nb_words, len(self.word_index))+1),
                                    als.EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=max_sequence_length,
                                    trainable=False)
        lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

        sequence_1_input = Input(shape=(max_sequence_length,), dtype='int32')
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        merged = lstm_layer(embedded_sequences_1)

        merged = Dropout(rate_drop_dense)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(num_dense, activation=act)(merged)
        merged = Dropout(rate_drop_dense)(merged)
        merged = BatchNormalization()(merged)

        preds = Dense(category_num, activation='softmax')(merged)
        return Model(inputs=sequence_1_input, outputs=preds)

    def create_x_y_data(self, train_final, labels_final):
        """
        创建品牌预测相关数据
        """
        # 加载映射器
        with open(path + 'predict/model/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        train_sequences = tokenizer.texts_to_sequences(train_final)
        # 生成X和Y
        self.X = pad_sequences(train_sequences, maxlen=als.MAX_SEQUENCE_LENGTH)
        self.Y = np_utils.to_categorical(np.array(labels_final))
        # print('Shape of data tensor:', self.X.shape)
        # print('Shape of label tensor:', self.Y.shape)

    def train_brand_model(self):
        """
        训练品牌模型并保存
        """
        self.create_x_y_data(self.train_final, self.labels_final)

        brand_map = pd.read_csv(path + 'predict/map/brand_map.csv')
        model = self.define_brand_model_structure(als.MAX_NB_WORDS, als.EMBEDDING_DIM, als.MAX_SEQUENCE_LENGTH, als.NUM_LSTM,
                                                  als.RATE_DROP_LSTM, als.RATE_DROP_DENSE, als.NUM_DENSE, als.ACT, len(brand_map))
        # 设置随机种子
        np.random.seed(als.SEED)
        # 早期停止回调
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=als.VERBOSE, mode='min')
        # 只保存最佳权重
        mcp_save = ModelCheckpoint(path + 'predict/model/brand_model_weights.hdf5',
                                   save_best_only=True, monitor='val_loss', mode='min')
        # 模型编译
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        # 训练
        batch_size = cal_train_param(self.X)
        model.fit(self.X, self.Y, epochs=als.EPOCHS, batch_size=batch_size, shuffle=True, verbose=als.VERBOSE,
                  callbacks=[early_stopping, mcp_save], validation_split=0.25)

    def train_model_model(self):
        """
        训练车型模型并保存
        """
        model_map = pd.read_csv(path + 'predict/map/model_map.csv')
        model_map = model_map.loc[(model_map['brand_slug'].notnull()), :]
        model_map.reset_index(inplace=True, drop=True)
        exception_model_predict = []

        # for i, brand_slug in enumerate(list(set(model_map.brand_slug.values))):
        for i, brand_slug in enumerate(['dazhong']):
            print(i, 'start model train:', brand_slug)
            # 定位品牌
            train = self.train.loc[(self.train['brand_slug'] == brand_slug), :]
            train.reset_index(inplace=True, drop=True)
            # 生成车型映射表
            car_model = pd.DataFrame(pd.Series(list(set(train.global_slug.values))), columns=['global_slug'])
            car_model = car_model.reset_index(drop=True).reset_index()
            car_model = car_model.rename(columns={'index': 'model_id'})
            os.makedirs(os.path.dirname(path + 'predict/model/brand/' + brand_slug + '/model_map.csv'), exist_ok=True)
            car_model.to_csv(path + 'predict/model/brand/' + brand_slug + '/model_map.csv', index=False)
            car_model = car_model.loc[:, ['global_slug', 'model_id']]
            train = train.merge(car_model, how='left', on=['global_slug'])
            if len(car_model) == 1:
                print('finish brand train:', brand_slug)
                continue
            # 生成训练和标签数据
            self.combine_train_label(train, 'model_id')
            self.create_x_y_data(self.train_final, self.labels_final)
            # 定义模型并训练
            model = self.define_brand_model_structure(als.MAX_NB_WORDS, als.EMBEDDING_DIM, als.MAX_SEQUENCE_LENGTH,
                                                      als.NUM_LSTM,
                                                      als.RATE_DROP_LSTM, als.RATE_DROP_DENSE, als.NUM_DENSE, als.ACT,
                                                      len(list(set(train.global_slug.values))))
            # 设置随机种子
            np.random.seed(als.SEED)
            # 早期停止回调
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=als.VERBOSE, mode='min')
            # 只保存最佳权重
            mcp_save = ModelCheckpoint(path + 'predict/model/brand/' + brand_slug + '/model_model_weights.hdf5', save_best_only=True, monitor='val_loss', mode='min')
            # 模型编译
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
            # 训练
            batch_size = cal_train_param(self.X)
            model.fit(self.X, self.Y, epochs=als.EPOCHS, batch_size=batch_size, shuffle=True, verbose=als.VERBOSE,
                      callbacks=[early_stopping, mcp_save], validation_split=0.25)

            # 训练精度低于阈值的异常品牌
            score, acc = model.evaluate(self.X, self.Y, batch_size=batch_size, verbose=als.VERBOSE)
            if acc < als.THRESHOLD_MODEL:
                exception_model_predict.append(brand_slug)
            print(i, 'finish brand train:', brand_slug)
        exception_brand = pd.DataFrame(exception_model_predict, columns=['exception_brand'])
        exception_brand.to_csv(path + 'predict/model/brand/exception_model_predict.csv', index=False)

    def train_details_model(self):
        """
        训练款型模型并保存
        """
        detail_map = pd.read_csv(path + 'predict/map/detail_map.csv')
        detail_map = detail_map.loc[(detail_map['brand_slug'].notnull()), :]
        detail_map.reset_index(inplace=True, drop=True)
        exception_model_predict = []

        # for i, brand_slug in enumerate(list(set(detail_map.brand_slug.values))):
        for i, brand_slug in enumerate(['dazhong']):
            print(i, 'start model train:', brand_slug)
            # 定位品牌
            train = self.train.loc[(self.train['brand_slug'] == brand_slug), :]
            train.reset_index(inplace=True, drop=True)
            # 生成车型映射表
            car_details = pd.DataFrame(pd.Series(list(set(train.model_detail_slug.values))), columns=['model_detail_slug'])
            car_details = car_details.reset_index(drop=True).reset_index()
            car_details = car_details.rename(columns={'index': 'detail_id'})
            os.makedirs(os.path.dirname(path + 'predict/model/brand/' + brand_slug + '/detail_map.csv'), exist_ok=True)
            car_details.to_csv(path + 'predict/model/brand/' + brand_slug + '/detail_map.csv', index=False)
            car_details = car_details.loc[:, ['model_detail_slug', 'detail_id']]
            train = train.merge(car_details, how='left', on=['model_detail_slug'])
            if len(car_details) == 1:
                print('finish model train:', brand_slug)
                continue
            # 生成训练和标签数据
            self.combine_train_label(train, 'detail_id')
            self.create_x_y_data(self.train_final, self.labels_final)
            # 定义模型并训练
            model = self.define_brand_model_structure(als.MAX_NB_WORDS, als.EMBEDDING_DIM, als.MAX_SEQUENCE_LENGTH,
                                                      als.NUM_LSTM,
                                                      als.RATE_DROP_LSTM, als.RATE_DROP_DENSE, als.NUM_DENSE, als.ACT,
                                                      len(list(set(train.model_detail_slug.values))))
            # 设置随机种子
            np.random.seed(als.SEED)
            # 早期停止回调
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=als.VERBOSE, mode='min')
            # 只保存最佳权重
            mcp_save = ModelCheckpoint(path + 'predict/model/brand/' + brand_slug + '/detail_model_weights.hdf5', save_best_only=True, monitor='val_loss', mode='min')
            # 模型编译
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
            # 训练
            batch_size = cal_train_param(self.X)
            model.fit(self.X, self.Y, epochs=als.EPOCHS, batch_size=batch_size, shuffle=True, verbose=als.VERBOSE,
                      callbacks=[early_stopping, mcp_save], validation_split=0.25)

            # 训练精度低于阈值的异常车型
            score, acc = model.evaluate(self.X, self.Y, batch_size=batch_size, verbose=als.VERBOSE)
            if acc < als.THRESHOLD_DETAIL:
                exception_model_predict.append(brand_slug)
            print(i, 'finish model train:', brand_slug)
        exception_brand = pd.DataFrame(exception_model_predict, columns=['exception_brand'])
        exception_brand.to_csv(path + 'predict/model/brand/exception_detail_predict.csv', index=False)

    def execute(self):
        """
        执行模型训练
        """
        try:
            self.init_variable()
            self.define_predict_map()
            self.combine_train_label(self.train, 'brand_id')
            self.create_tokenizer()
            # 训练品牌预测模型
            # self.train_brand_model()
            # # 训练车型预测模型
            self.train_model_model()
            # # 训练款型预测模型
            self.train_details_model()
        except Exception:
            raise StackingTrainError(traceback.format_exc())

