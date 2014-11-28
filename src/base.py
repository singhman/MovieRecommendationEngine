import numpy as np
from scipy import sparse
import inspect
import warnings

def iteritems(d, **kw):
    return iter(getattr(d, "iteritems")(**kw))

class BaseEstimator(object):
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        try:
            init = getattr(cls.__init__, 'deprecated_original', cls.__init__)

            # introspect the constructor arguments to find the model parameters
            # to represent
            args, varargs, kw, default = inspect.getargspec(init)
            if not varargs is None:
                raise RuntimeError("estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s doesn't follow this convention."
                                   % (cls, ))
            args.pop(0)
        except TypeError:
            # No explicit __init__
            args = []
        args.sort()
        return args

    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    continue
            finally:
                warnings.filters.pop(0)

            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if not name in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if not key in valid_params:
                    raise ValueError('Invalid parameter %s ' 'for estimator %s'
                                     % (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name),),)

    def __str__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name,
                           _pprint(self.get_params(deep=True),
                                   offset=len(class_name), printer=str,),)
