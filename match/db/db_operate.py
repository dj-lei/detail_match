from match.db import *


def query_produce_open_model_detail():
    """
    查询款型库
    """
    query_sql = 'select * from open_model_detail'

    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_produce_open_category():
    """
    查询车型库
    """
    query_sql = 'select * from open_category '

    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_produce_model_detail_normal():
    """
    查询车型库
    """
    query_sql = 'select id,brand,model,model_detail,global_name,global_slug,year,volume,model_detail_slug_id,domain,status ' \
                'from model_detail_normal where status = \'B\' or status = \'M\''

    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_model_product_review_source_data(start_time):
    """
    查询生产库交易训练数据
    """
    query_sql = 'select id,title,brand_slug,model_slug,model_detail_slug ' \
                        'from car_source where status = \'review\' and global_sibling = 0 and sold_time >= \''+start_time+'\' '

    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_model_product_sale_source_data(start_time):
    """
    查询生产库在售训练数据
    """
    query_sql = 'select id,title,brand_slug,model_slug,model_detail_slug ' \
                        'from car_source where status = \'sale\' and global_sibling = 0 and pub_time >= \''+start_time+'\' '

    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_model_product_source_data(start_time, end_time):
    """
    查询生产库交易训练数据
    """
    query_sql = 'select cs.id,cs.title,cs.brand_slug,cs.model_slug,cs.model_detail_slug,cs.domain from pingjia.car_source as cs ' \
                'where model_detail_slug is not null and global_sibling = 0 and cs.pub_time >= \''+start_time+'\' and cs.pub_time <= \''+ end_time +'\' '

    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_spider_update_product_source_data(start_time, end_time):
    """
    查询爬虫更新过的训练数据
    """
    query_sql = 'select cs.id,cs.title,cs.brand_slug,cs.model_slug,cs.model_detail_slug,cdi.mdn_status from pingjia.car_source as cs ' \
                'left join pingjia.car_detail_info as cdi on cs.id = cdi.car_id ' \
                'left join pingjia.product_update_history as puh on cs.pid =  puh.pid ' \
                'where puh.update_time >= \''+start_time+'\' and puh.update_time <= \''+ end_time +'\' '

    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def update_match(data):
    """
    查询生产库交易训练数据
    """
    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE)
    with engine.begin() as con:
        for i in range(0, len(data)):
            car_id = data.loc[i, 'id']
            brand_slug = data.loc[i, 'brand_slug']
            brand_slug = 'null' if str(brand_slug) == 'nan' else brand_slug
            model_slug = data.loc[i, 'model_slug']
            model_slug = 'null' if str(model_slug) == 'nan' else model_slug
            model_detail_slug = data.loc[i, 'model_detail_slug']
            model_detail_slug = 'null' if str(model_detail_slug) == 'nan' else model_detail_slug
            mdn_status = data.loc[i, 'mdn_status']
            print('update:', car_id, brand_slug, model_slug, model_detail_slug, mdn_status)
            con.execute("""
               UPDATE car_source as cs,car_detail_info as cdi
               set cs.brand_slug=%s,cs.model_slug=%s,cs.model_detail_slug=%s,cdi.mdn_status=%s
               WHERE cs.id=cdi.car_id and cs.id=%s
            """, (brand_slug, model_slug, model_detail_slug, mdn_status, str(car_id)))
    con.close()


def insert_valuate_detail_match_error(data):
    """
    插入款型匹配不一致数据
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)

    data.to_sql(name='valuate_detail_match_error', if_exists='append', con=engine, index=False)


def insert_valuate_error_info(e):
    """
    存储异常情况
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    error_type = e.error_type
    model_slug = gl.ERROR_MATCH
    message = e.message
    state = 'unprocessed'
    create_time = datetime.datetime.now()
    with engine.begin() as con:
        con.execute("""
           INSERT INTO valuate_error_history (error_type, model_slug, description, state, create_time) VALUES (%s, %s, %s, %s, %s)
        """, (error_type, model_slug, message, state, create_time))
    con.close()