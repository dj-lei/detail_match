from match.db import *


def insert_or_update_match_cos_vector(data):
    """
    插入或更新
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)

    with engine.begin() as con:
        sql = 'TRUNCATE TABLE china_used_car_estimate.match_cos_vector'
        con.execute(sql)
    con.close()

    data.to_sql(name='match_cos_vector', if_exists='append', con=engine, index=False)


def insert_or_update_match_brand_name(data):
    """
    插入或更新
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)

    with engine.begin() as con:
        sql = 'TRUNCATE TABLE china_used_car_estimate.match_brand_name'
        con.execute(sql)
    con.close()

    data.to_sql(name='match_brand_name', if_exists='append', con=engine, index=False)


def insert_or_update_match_word_index(data):
    """
    插入或更新
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)

    with engine.begin() as con:
        sql = 'TRUNCATE TABLE china_used_car_estimate.match_word_index'
        con.execute(sql)
    con.close()

    data.to_sql(name='match_word_index', if_exists='append', con=engine, index=False)


def query_car_source():
    """
    插入或更新
    """
    start_time = datetime.datetime.now() - datetime.timedelta(days=30)
    start_time = start_time.strftime("%Y-%m-%d") + ' 00:00:00'
    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)

    query_sql = 'select url,brand_name,model_name,detail_name,year,month,mile,city,price,domain,labels,create_time,deal_date_ts from crawler_car_source where transfer_owner = 0 and domain in (\'guazi.com\',\'ttpai.cn\',\'xin.com\') and create_time >= \'' + start_time + '\''
    return pd.read_sql_query(query_sql, engine)


def query_product_open_category():
    """
    插入或更新
    """
    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)

    query_sql = 'select * from pingjia.open_category where status = \'Y\' '
    return pd.read_sql_query(query_sql, engine)


def query_product_open_model_detail():
    """
    插入或更新
    """
    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)

    query_sql = 'select * from pingjia.open_model_detail where status = \'Y\' '
    return pd.read_sql_query(query_sql, engine)