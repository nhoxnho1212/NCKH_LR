import os
import logging
log = logging.getLogger('root')
def init(dirName):
    try:
        os.mkdir(f'{dirName}')
        log.info(f'Directory {dirName} is created' )
    except FileExistsError:
        log.warning(f'Directory {dirName} already exists' )
    except Exception as e:
        log.error(e)
