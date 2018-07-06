import json
import numpy as np
import pandas as pd


class DataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '__type__': '__np.ndarray__',
                'content': obj.tolist()
            }
        elif isinstance(obj, pd.DataFrame):
            return {
                '__type__': '__pd.DataFrame__',
                'content': obj.to_json()
            }
        elif isinstance(obj, pd.Series):
            return {
                '__type__': '__pd.Series__',
                'content': obj.to_json()
            }
        else:
            return json.JSONEncoder.default(self, obj)


def data_decoder(obj):
    if '__type__' in obj:
        if obj['__type__'] == '__np.ndarray__':
            return np.array(obj['content'])
        elif obj['__type__'] == '__pd.DataFrame__':
            return pd.read_json(obj['content'], typ='frame')
        elif obj['__type__'] == '__pd.DataFrame__':
            return pd.read_json(obj['content'], typ='series')
    return obj


def data_dumps(obj):
    return json.dumps(obj, cls=DataEncoder)


def data_loads(obj):
    return json.loads(obj, object_hook=data_decoder)
