ENCODING = 'utf-8'

##########################
# 生产,测试库配置
##########################

# 运行环境[PRODUCT,TEST,LOCAL]
RUNTIME_ENVIRONMENT = 'LOCAL'

if RUNTIME_ENVIRONMENT == 'LOCAL':
    # 生产库外网
    PRODUCE_DB_ADDR_OUTTER = '59.110.233.103'
    PRODUCE_DB_USER = 'pingjia'
    PRODUCE_DB_PASSWD = 'De32wsxC'
    PRODUCE_PINGJIA_ENGINE = 'mysql+pymysql://'+PRODUCE_DB_USER+':'+PRODUCE_DB_PASSWD+'@'+PRODUCE_DB_ADDR_OUTTER+'/china_used_car_estimate?charset=utf8'

    # 测试库
    TEST_DB_ADDR = '192.168.2.114'
    TEST_DB_USER = 'pingjia'
    TEST_DB_PASSWD = 'De32wsxC'
    TEST_PINGJIA_ENGINE = 'mysql+pymysql://'+TEST_DB_USER+':'+TEST_DB_PASSWD+'@'+TEST_DB_ADDR+'/china_used_car_estimate?charset=utf8'

elif RUNTIME_ENVIRONMENT == 'PRODUCT':
    # 生产库外网
    PRODUCE_DB_ADDR_OUTTER = '192.168.2.114'
    PRODUCE_DB_USER = 'pingjia'
    PRODUCE_DB_PASSWD = 'De32wsxC'
    PRODUCE_PINGJIA_ENGINE = 'mysql+pymysql://' + PRODUCE_DB_USER + ':' + PRODUCE_DB_PASSWD + '@' + PRODUCE_DB_ADDR_OUTTER + '/china_used_car_estimate?charset=utf8'

    # 生产库
    TEST_DB_ADDR = '192.168.2.114'
    TEST_DB_USER = 'pingjia'
    TEST_DB_PASSWD = 'De32wsxC'
    TEST_PINGJIA_ENGINE = 'mysql+pymysql://' + TEST_DB_USER + ':' + TEST_DB_PASSWD + '@' + TEST_DB_ADDR + '/china_used_car_estimate?charset=utf8'

###########################
# 异常类型
###########################
ERROR_SQL = 'SQL'
ERROR_PARAMS = 'PARAMS'

ENGLISH_BRAND = ['mg', 'smart', 'mini', 'localmotors', 'acschnitzer', 'polestar', 'lorinser', 'alpina', 'jeep', 'lite', 'wey', 'ktm', 'gmc', 'ds']
