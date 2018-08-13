import json
from typing import Any

import numpy as np
import pandas as pd


class DataEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        Serialize numpy or pandas objects to json.
        """
        if isinstance(obj, np.ndarray):
            return {
                '__type__': '__np.ndarray__',
                '__content__': obj.tolist()
            }
        elif isinstance(obj, pd.DataFrame):
            return {
                '__type__': '__pd.DataFrame__',
                '__content__': obj.to_dict(orient='split')
            }
        elif isinstance(obj, pd.Series):
            return {
                '__type__': '__pd.Series__',
                '__content__': {
                    'dtype': str(obj.dtype),
                    'index': list(obj.index),
                    'data': obj.tolist(),
                    'name': obj.name
                }
            }
        else:
            return json.JSONEncoder.default(self, obj)


def data_decoder(obj):
    if '__type__' in obj:
        if obj['__type__'] == '__np.ndarray__':
            return np.array(obj['__content__'])
        elif obj['__type__'] == '__pd.DataFrame__':
            return pd.DataFrame(**obj['__content__'])
        elif obj['__type__'] == '__pd.Series__':
            return pd.Series(**obj['__content__'])
    return obj


def data_dumps(obj: Any) -> str:
    return json.dumps(obj, cls=DataEncoder)


def data_loads(obj: str) -> Any:
    return json.loads(obj, object_hook=data_decoder)
