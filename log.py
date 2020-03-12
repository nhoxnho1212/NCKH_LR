import logging
from constants import FILE_LOG

def setup_custom_logger(name):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(f'%(asctime)s : %(levelname)s : %(module)s : %(message)s')
    logger.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(FILE_LOG)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    return logger
