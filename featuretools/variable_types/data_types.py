from builtins import object
from datetime import date, time, timedelta, datetime

import numpy as np
import pandas as pd
from past.builtins import basestring

from featuretools.entityset.timedelta import Timedelta


class DataTypesMeta(type):
    _all = 'all'
    dtype_mapping = {
        'category': ('category',),
        'object': (object, np.object_),
        'string': (basestring, np.string_, np.unicode_),
        'bool': (bool, np.bool),
        'datetime': (datetime, date, time, np.datetime64),
        'timedelta': (Timedelta, timedelta, np.timedelta64),
        'integer': (int, np.integer),
        'float': (float, np.floating)
    }

    def __getattr__(cls, attr):
        if attr in cls.dtype_mapping:
            return cls.dtype_mapping[attr]
        elif attr == 'numeric':
            integers = cls.dtype_mapping['integer']
            floats = cls.dtype_mapping['float']
            return tuple(list(integers) + list(floats))
        else:
            raise AttributeError

    defaults = {
        'datetime': pd.Timestamp.now(),
        'integer': 0,
        'float': 0.1,
        'timedelta': pd.Timedelta('1d'),
        'object': 'object',
        'bool': True,
        'string': 'test',
    }

    def get_default(cls, dtype):
        for dt_str, dt_list in cls.dtype_mapping.items():
            if dt_str not in ['category', 'object'] and issubclass(dtype, dt_list):
                return cls.defaults[dt_str]
        return cls.defaults['object']

    def issubclass(cls, dtype, other_dtypes):
        if not isinstance(other_dtypes, (list, tuple)):
            other_dtypes = [other_dtypes]
        for dt in other_dtypes:
            if isinstance(dt, (tuple, list)):
                return any(cls.issubclass(dtype, _dt) for _dt in dt)
            if not isinstance(dtype, np.dtype):
                if not issubclass(dt, np.generic) and issubclass(dtype, dt):
                    return True
                continue
            if isinstance(dtype, np.dtype):
                if issubclass(dt, np.generic) and np.issubdtype(dtype, dt):
                    return True
                continue
            if issubclass(dtype, np.generic) and issubclass(dt, np.generic):
                if issubclass(dtype, dt):
                    return True
        return False


class DataTypes(metaclass=DataTypesMeta):
    pass
