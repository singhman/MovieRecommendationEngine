import numpy as np
from scipy import sparse
import operator
import copy
import inspect
from datamodel import UserNotFoundError, ItemNotFoundError
from recommender import UserBasedRecommender, ItemBasedRecommender
from metrics import root_mean_square_error
from metrics import mean_absolute_error
from metrics import normalized_mean_absolute_error
from metrics import evaluation_error
from sampling import SplitSampling

evaluation_metrics = {
    'rmse': root_mean_square_error,
    'mae': mean_absolute_error,
    'nmae': normalized_mean_absolute_error,
}

def iteritems(d, **kw):
    return iter(getattr(d, "iteritems")(**kw))

def check_sampling(sampling, n):
    if sampling is None:
        sampling = 1.0
    if operator.isNumberType(sampling):
        sampling = SplitSampling(n, evaluation_fraction=sampling)

    return sampling

def clone(estimator, safe=True):
    estimator_type = type(estimator)
    if estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, 'get_params'):
        if not safe:
            return copy.deepcopy(estimator)
        else:
            raise TypeError("Cannot clone object '%s' (type %s): "
                            "it does not seem to be a base estimator a"
                            " it does not implement a 'get_params' methods."
                            % (repr(estimator), type(estimator)))
    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in iteritems(new_object_params):
        new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)

    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if isinstance(param1, np.ndarray):
            # For most ndarrays, we do not test for complete equality
            if not isinstance(param2, type(param1)):
                equality_test = False
            elif (param1.ndim > 0
                    and param1.shape[0] > 0
                    and isinstance(param2, np.ndarray)
                    and param2.ndim > 0
                    and param2.shape[0] > 0):
                equality_test = (
                    param1.shape == param2.shape
                    and param1.dtype == param2.dtype
                    # We have to use '.flat' for 2D arrays
                    and param1.flat[0] == param2.flat[0]
                    and param1.flat[-1] == param2.flat[-1]
                )
            else:
                equality_test = np.all(param1 == param2)
        elif sparse.issparse(param1):
            # For sparse matrices equality doesn't work
            if not sparse.issparse(param2):
                equality_test = False
            elif param1.size == 0 or param2.size == 0:
                equality_test = (
                    param1.__class__ == param2.__class__
                    and param1.size == 0
                    and param2.size == 0
                )
            else:
                equality_test = (
                    param1.__class__ == param2.__class__
                    and param1.data[0] == param2.data[0]
                    and param1.data[-1] == param2.data[-1]
                    and param1.nnz == param2.nnz
                    and param1.shape == param2.shape
                )
        else:
            equality_test = new_object_params[name] == params_set[name]
        if not equality_test:
            raise RuntimeError('Cannot clone object %s, as the constructor '
                               'does not seem to set parameter %s' %
                               (estimator, name))

    return new_object

class Evaluator(object):
    def _build_recommender(self, dataset, recommender):
        recommender_training = clone(recommender)
        if not recommender.model.has_preference_values():
            recommender_training.model.dataset = \
                    recommender_training.model._load_dataset(dataset.copy())
        else:
            recommender_training.model.dataset = dataset

        if hasattr(recommender_training.model, 'build_model'):
            recommender_training.model.build_model()

        return recommender_training

    def evaluate(self, recommender, metric=None, **kwargs):
        sampling_users = kwargs.pop('sampling_users', None)
        sampling_ratings = kwargs.pop('sampling_ratings', 0.7)
        permutation = kwargs.pop('permutation', True)
        at = kwargs.pop('at', 3)

        if metric not in evaluation_metrics and metric is not None:
            raise ValueError('metric %s is not recognized. valid keywords \
              are %s' % (metric, evaluation_metrics.keys()))

        n_users = recommender.model.users_count()
        sampling_users = check_sampling(sampling_users, n_users)
        users_set, _ = sampling_users.split(permutation=permutation)

        training_set = {}
        testing_set = {}

        #Select the users to be evaluated.
        user_ids = recommender.model.user_ids()
        for user_id in user_ids[users_set]:
            #Select the ratings to be evaluated.
            preferences = recommender.model.preferences_from_user(user_id)
            sampling_eval = check_sampling(sampling_ratings, len(preferences))
            train_set, test_set = sampling_eval.split(indices=True,permutation=permutation)

            preferences = list(preferences)
            if recommender.model.has_preference_values():
                training_set[user_id] = dict((preferences[idx]
                             for idx in train_set)) if preferences else {}
                testing_set[user_id] = [preferences[idx]
                             for idx in test_set] if preferences else []
            else:
                training_set[user_id] = dict(((preferences[idx], 1.0)
                             for idx in train_set)) if preferences else {}
                testing_set[user_id] = [(preferences[idx], 1.0)
                             for idx in test_set] if preferences else []

        #Evaluate the recommender.
        recommender_training = self._build_recommender(training_set, recommender)

        real_preferences = []
        estimated_preferences = []

        if isinstance(recommender, UserBasedRecommender):
            for user_id, preferences in testing_set.iteritems():
                print 'Evaluating user {0}'.format(user_id)
                nearest_neighbors, similarities, user_pref_mean, neighbors_mean = recommender_training.user_sim_neighbors(user_id)
                for item_id, preference in preferences:
                    #Estimate the preferences
                    try:
                        #estimated = recommender_training._estimate_preference(user_id, item_id)
                        estimated = recommender_training.estimate_preference(user_id, item_id, nearest_neighbors, similarities, user_pref_mean, neighbors_mean)
                        real_preferences.append(preference)
                        print 'Estimated:{0}\tReal:{1}'.format(estimated,preference)
                    except ItemNotFoundError:
                        # It is possible that an item exists in the test data but
                        # not training data in which case an exception will be
                        # throw. Just ignore it and move on
                        continue
                    estimated_preferences.append(estimated)
        elif isinstance(recommender, ItemBasedRecommender):
            for user_id, preferences in testing_set.iteritems():
                print 'Evaluating user {0}'.format(user_id)
                for item_id, preference in preferences:
                    #Estimate the preferences
                    try:
                        estimated = recommender_training.estimate_preference(user_id, item_id)
                        real_preferences.append(preference)
                    except:
                        continue
                    estimated_preferences.append(estimated)
        print 'estimated_preferences:{0}\nReal preferences:{1}'.format(estimated_preferences, real_preferences)
        real_preferences = np.array(real_preferences)
        estimated_preferences = np.array(estimated_preferences)
        real_preferences = real_preferences[~np.isnan(estimated_preferences)]
        estimated_preferences = estimated_preferences[~np.isnan(estimated_preferences)]

        #Return the error results.
        if metric in ['rmse', 'mae', 'nmae']:
            eval_function = evaluation_metrics[metric]
            if metric == 'nmae':
                return {metric: eval_function(real_preferences,
                                          estimated_preferences,
                                recommender.model.maximum_preference_value(),
                                recommender.model.minimum_preference_value())}
            return {metric: eval_function(real_preferences,
                                          estimated_preferences)}
        return evaluation_error(real_preferences,
                    estimated_preferences,
                    recommender.model.maximum_preference_value(),
                    recommender.model.minimum_preference_value())
