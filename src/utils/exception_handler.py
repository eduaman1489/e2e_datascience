import sys

def error_details_capture(error_message,error_details):
    _,_,exc_tb=error_details.exc_info() # First 2 are not required, just need execution traceback
    lineno=exc_tb.tb_lineno
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message =  f" \n Error occured in script name : {file_name} \n Line number : {lineno} \n Error message is :{str(error_message)}"
    return error_message

class Custom_Exception(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message=error_details_capture(error_message,error_details)

    def __str__(self):
        return self.error_message
    
# if __name__=="__main__":
#     try:
#         a=1/0
#     except Exception as e:
#         raise Custom_Exception(e,sys)