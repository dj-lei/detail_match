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
    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)

    query_sql = 'select url,detail_name,year,month,mile,city,price,domain,labels,create_time from crawler_car_source where domain = \'guazi.com\' '
    return pd.read_sql_query(query_sql, engine)