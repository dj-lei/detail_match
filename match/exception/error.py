from match.exception import *


class SqlOperateError(Exception):
    """
    操作数据库异常
    """
    def __init__(self):
        self.error_type = gl.ERROR_SQL
        self.message = traceback.format_exc()


class ApiParamsError(Exception):
    """
    Api参数类型异常
    """
    def __init__(self, message):
        self.error_type = gl.ERROR_PARAMS
        self.message = message


