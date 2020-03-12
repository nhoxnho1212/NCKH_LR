import logging
log = logging.getLogger('root')

try:
    import pandas as pd
except Exception as e:
    log.critical(e)
    exit(0)

def oneHotProcessing(field_name,df):
    temp_data=df
    try:
        values = pd.get_dummies(df[field_name],prefix=field_name)
        return pd.concat([df.drop(columns=[field_name]),values],axis =1)
    except Exception:
        print(Exception)
        return temp_data