import logging
import os
from datetime import datetime

# creating file with date and time name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# Joinning log path with file
lOG_PATH = os.path.join(os.getcwd(),"logs",LOG_FILE)
# creating dir
os.makedirs(lOG_PATH,exist_ok=True)
# joinning log path with log file in order to save all log file under log folder
LOG_FILE_PATH = os.path.join(lOG_PATH,LOG_FILE)
# basicConfig
logging.basicConfig(
    filename = LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    
)

