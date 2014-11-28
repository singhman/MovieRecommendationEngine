import numpy as np
import scipy.sparse as sp

def root_mean_square_error(y_real, y_pred):
    y_real, y_pred = check_arrays(y_real, y_pred)
    return np.sqrt((np.sum((y_pred - y_real) ** 2)) / y_real.shape[0])


def mean_absolute_error(y_real, y_pred):
    y_real, y_pred = check_arrays(y_real, y_pred)
    return np.sum(np.abs(y_pred - y_real)) / y_real.size


def normalized_mean_absolute_error(y_real, y_pred, max_rating, min_rating):
    y_real, y_pred = check_arrays(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    return mae / (max_rating - min_rating)


def evaluation_error(y_real, y_pred, max_rating, min_rating):
    """
    Returns
    -------
    mae: Positive floating point value: the best value is 0.0.
    nmae: Positive floating point value: the best value is 0.0.
    rmse: Positive floating point value: the best value is 0.0.
    """
    mae = mean_absolute_error(y_real, y_pred)
    nmae = normalized_mean_absolute_error(y_real, y_pred,
             max_rating, min_rating)
    rmse = root_mean_square_error(y_real, y_pred)

    return mae, nmae, rmse

def check_arrays(*arrays, **options):
    sparse_format = options.pop('sparse_format', None)
    if sparse_format not in (None, 'csr', 'csc'):
        raise ValueError('Unexpected sparse format: %r' % sparse_format)
    copy = options.pop('copy', False)
    if options:
        raise ValueError("Unexpected kw arguments: %r" % options.keys())

    if len(arrays) == 0:
        return None

    first = arrays[0]
    if not hasattr(first, '__len__') and not hasattr(first, 'shape'):
        raise ValueError("Expected python sequence or array, got %r" % first)
    n_samples = first.shape[0] if hasattr(first, 'shape') else len(first)

    checked_arrays = []
    for array in arrays:
        array_orig = array
        if array is None:
            # special case: ignore optional y=None kwarg pattern
            checked_arrays.append(array)
            continue

        if not hasattr(array, '__len__') and not hasattr(array, 'shape'):
            raise ValueError("Expected python sequence or array, got %r"
                             % array)
        size = array.shape[0] if hasattr(array, 'shape') else len(array)

        if size != n_samples:
            raise ValueError("Found array with dim %d. Expected %d" % (
                size, n_samples))

        if sp.issparse(array):
            if sparse_format == 'csr':
                array = array.tocsr()
            elif sparse_format == 'csc':
                array = array.tocsc()
        else:
            array = np.asanyarray(array)

        if copy and array is array_orig:
            array = array.copy()
        checked_arrays.append(array)

    return checked_arrays
