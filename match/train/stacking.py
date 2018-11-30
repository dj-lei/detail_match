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

    def init_variable(self):
        """
        初始化变量
        """
        self.train = pd.read_csv(path + '../tmp/train/train_final.csv')
        self.word2vec = Word2Vec.load(path + 'predict/model/word2vec.bin')

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

    def define_brand_model_structure(self, max_nb_words, embedding_dim, max_sequence_length, num_lstm,
                                     rate_drop_lstm, rate_drop_dense, num_dense, act, category_num):
        """
        定义模型结构
        """
        # 加载词典
        f = open(path + 'predict/model/word_index.txt', 'r', encoding='UTF-8')
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
        train = self.train.copy()
        # 生成品牌映射表
        car_brand = pd.DataFrame(pd.Series(list(set(train.brand_slug.values))), columns=['brand_slug'])
        car_brand = car_brand.reset_index(drop=True).reset_index()
        car_brand = car_brand.rename(columns={'index': 'brand_id'})
        os.makedirs(os.path.dirname(path + 'predict/model/brand_map.csv'), exist_ok=True)
        car_brand.to_csv(path + 'predict/model/brand_map.csv', index=False)
        car_brand = car_brand.loc[:, ['brand_slug', 'brand_id']]
        train = train.merge(car_brand, how='left', on=['brand_slug'])

        self.combine_train_label(train, 'brand_id')
        self.create_x_y_data(self.train_final, self.labels_final)
        brand_map = pd.read_csv(path + 'predict/map/brand_map.csv')
        model = self.define_brand_model_structure(als.MAX_NB_WORDS, als.EMBEDDING_DIM, als.MAX_SEQUENCE_LENGTH, als.NUM_LSTM,
                                                  als.RATE_DROP_LSTM, als.RATE_DROP_DENSE, als.NUM_DENSE, als.ACT, len(brand_map))
        # 设置随机种子
        np.random.seed(als.SEED)
        # 早期停止回调
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=als.VERBOSE, mode='min')
        # 只保存最佳权重
        mcp_save = ModelCheckpoint(path + 'predict/model/brand_model_weights.hdf5', save_best_only=True, monitor='val_loss', mode='min')
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
        brand_map = pd.read_csv(path + 'predict/map/brand_map.csv')
        exception_model_predict = []

        # for i, brand_slug in enumerate(list(set(brand_map.brand_slug.values))):
        for i, brand_slug in enumerate(['dazhong']):
            print(i, 'start model train:', brand_slug)
            # 定位品牌
            train = self.train.loc[(self.train['brand_slug'] == brand_slug), :]
            train.reset_index(inplace=True, drop=True)
            # 生成车型映射表
            car_model = pd.DataFrame(pd.Series(list(set(train.model_slug.values))), columns=['model_slug'])
            car_model = car_model.reset_index(drop=True).reset_index()
            car_model = car_model.rename(columns={'index': 'model_id'})
            os.makedirs(os.path.dirname(path + 'predict/model/brand/' + brand_slug + '/model_map.csv'), exist_ok=True)
            car_model.to_csv(path + 'predict/model/brand/' + brand_slug + '/model_map.csv', index=False)
            car_model = car_model.loc[:, ['model_slug', 'model_id']]
            train = train.merge(car_model, how='left', on=['model_slug'])
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
        brand_map = pd.read_csv(path + 'predict/map/brand_map.csv')
        exception_model_predict = []

        # for i, brand_slug in enumerate(list(set(brand_map.brand_slug.values))):
        for i, brand_slug in enumerate(['dazhong']):
            print(i, 'start model train:', brand_slug)
            # 定位品牌
            train = self.train.loc[(self.train['brand_slug'] == brand_slug), :]
            train.reset_index(inplace=True, drop=True)
            # 生成车型映射表
            car_details = pd.DataFrame(pd.Series(list(set(train.detail_slug.values))), columns=['detail_slug'])
            car_details = car_details.reset_index(drop=True).reset_index()
            car_details = car_details.rename(columns={'index': 'detail_id'})
            os.makedirs(os.path.dirname(path + 'predict/model/brand/' + brand_slug + '/detail_map.csv'), exist_ok=True)
            car_details.to_csv(path + 'predict/model/brand/' + brand_slug + '/detail_map.csv', index=False)
            car_details = car_details.loc[:, ['detail_slug', 'detail_id']]
            train = train.merge(car_details, how='left', on=['detail_slug'])
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
            # 训练品牌预测模型
            self.train_brand_model()
            # # 训练车型预测模型
            self.train_model_model()
            # # 训练款型预测模型
            self.train_details_model()
        except Exception:
            raise StackingTrainError(traceback.format_exc())

