import sys

def error_message_details(error,error_detail:sys):
    """
    Desc: this method help you trace where exceptions has occurred and in which file
          and which line.
    """
    _,_,exc_tb = error_detail.exc_info()
    # This will trace your file name like in which file you have error
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,
        exc_tb.tb_lineno,
        str(error)
    )

    return error_message

class CustomExceptions(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_detail)

    def __str__(self):
        return self.error_message