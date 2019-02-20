from match.db import *


def query_produce_crawler_car_source():
    """
    查询款型库
    """
    query_sql = 'select id,source_id,detail_name,year,month,mile,city,color,car_application,price,domain,labels,transfer_owner' \
                ',annual_insurance,compulsory_insurance,business_insurance,create_time from china_used_car_estimate.crawler_car_source'

    engine = create_engine(gl.TEST_CHINA_USED_CAR_ESTIMATE_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)
