ENCODING = 'utf-8'

##########################
# 生产,测试库配置
##########################

# 运行环境[PRODUCT,TEST,LOCAL]
RUNTIME_ENVIRONMENT = 'LOCAL'

if RUNTIME_ENVIRONMENT == 'LOCAL':
    # 生产库外网
    PRODUCE_DB_ADDR_OUTTER = '101.201.143.74'
    PRODUCE_DB_USER = 'leidengjun'
    PRODUCE_DB_PASSWD = 'ldj_DEV_~!'
    PRODUCE_PINGJIA_ENGINE = 'mysql+pymysql://' + PRODUCE_DB_USER + ':' + PRODUCE_DB_PASSWD + '@' + PRODUCE_DB_ADDR_OUTTER + '/pingjia?charset=utf8'
    PRODUCE_CHINA_USED_CAR_ESTIMATE_ENGINE = 'mysql+pymysql://' + PRODUCE_DB_USER + ':' + PRODUCE_DB_PASSWD + '@' + PRODUCE_DB_ADDR_OUTTER + '/china_used_car_estimate?charset=utf8'

    # 测试库
    TEST_DB_ADDR = '101.200.229.249'
    TEST_DB_USER = 'pingjia'
    TEST_DB_PASSWD = 'De32wsxC'
    TEST_PINGJIA_ENGINE = 'mysql+pymysql://'+TEST_DB_USER+':'+TEST_DB_PASSWD+'@'+TEST_DB_ADDR+'/valuate?charset=utf8'
    TEST_CHINA_USED_CAR_ESTIMATE_ENGINE = 'mysql+pymysql://' + TEST_DB_USER + ':' + TEST_DB_PASSWD + '@' + TEST_DB_ADDR + '/china_used_car_estimate?charset=utf8'

elif RUNTIME_ENVIRONMENT == 'PRODUCT':
    # 生产库内网
    PRODUCE_DB_ADDR_OUTTER = '10.45.138.200'
    PRODUCE_DB_USER = 'leidengjun'
    PRODUCE_DB_PASSWD = 'ldj_DEV_~!'
    PRODUCE_PINGJIA_ENGINE = 'mysql+pymysql://' + PRODUCE_DB_USER + ':' + PRODUCE_DB_PASSWD + '@' + PRODUCE_DB_ADDR_OUTTER + '/pingjia?charset=utf8'
    PRODUCE_CHINA_USED_CAR_ESTIMATE_ENGINE = 'mysql+pymysql://' + PRODUCE_DB_USER + ':' + PRODUCE_DB_PASSWD + '@' + PRODUCE_DB_ADDR_OUTTER + '/china_used_car_estimate?charset=utf8'
    # 生产库
    TEST_DB_ADDR = '10.174.11.103:3306'
    TEST_DB_USER = 'leidengjun'
    TEST_DB_PASSWD = 'ldj_DEV_~!'
    TEST_PINGJIA_ENGINE = 'mysql+pymysql://' + TEST_DB_USER + ':' + TEST_DB_PASSWD + '@' + TEST_DB_ADDR + '/valuate?charset=utf8'

###########################
# 异常类型
###########################
ERROR_SQL = 'SQL'
ERROR_FE = 'FE'
ERROR_TRAIN = 'TRAIN'
ERROR_CRON = 'CRON'
ERROR_PREDICT_MODEL = 'PREDICT'
ERROR_MATCH = 'MATCH'

