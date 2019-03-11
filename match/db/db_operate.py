from match.db import *


def query_produce_crawler_car_source():
    """
    查询款型库
    """
    # query_sql = 'select id,source_id,detail_name,url,year,month,mile,city,color,car_application,price,domain,labels,transfer_owner,phone,thumbnail' \
    #             ',annual_insurance,compulsory_insurance,business_insurance,create_time from china_used_car_estimate.crawler_car_source ' \
    #             'where create_time > \'2019-03-04 09:00:00\' and  create_time < \'2019-03-04 17:30:00\' '

    query_sql = 'select * from china_used_car_estimate.crawler_car_source ' \
                'where create_time > \'2019-03-04 09:00:00\' and  create_time < \'2019-03-04 19:00:00\' '
    engine = create_engine(gl.PRODUCE_CHINA_USED_CAR_ESTIMATE_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)
