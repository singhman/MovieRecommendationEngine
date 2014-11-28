import numpy as np
from similarities import UserSimilarity
from distances import pearson_correlation

class BaseUserNeighborhoodStrategy(object):
    def user_neighborhood(self, user_id, data_model, n_similarity='user_similarity',
                distance=None, n_users=None, **params):
        raise NotImplementedError("BaseCandidateItemsStrategy is an abstract class.")

class AllNeighborsStrategy(object):
    '''
    Returns
    --------
    Returns all users in the model.
    This strategy is not recommended for large datasets and it is the dummiest one.
    '''
    def user_neighborhood(self, user_id, data_model, similarity='user_similarity',
        distance=None, nhood_size=None, **params):
        user_ids = data_model.user_ids()
        return user_ids[user_ids != user_id] if user_ids.size else user_ids

class NearestNeighborsStrategy(BaseUserNeighborhoodStrategy):
    '''
    Returns
    --------
    Returns the neighborhood consisting of the nearest n
    users to a given user. "Nearest" in this context is
    defined by the Similarity.
    '''
    def __init__(self):
        self.similarity = None

    def _set_similarity(self, data_model, similarity, distance, nhood_size):
        if not isinstance(self.similarity, UserSimilarity) \
             or not distance == self.similarity.distance:
            nhood_size = nhood_size if not nhood_size else nhood_size + 1
            self.similarity = UserSimilarity(data_model, distance, nhood_size)

    def user_neighborhood(self, user_id, data_model, n_similarity='user_similarity',
             distance=None, nhood_size=None, **params):

        minimal_similarity = params.get('minimal_similarity', 0.5)

        #set the nhood_size at Similarity , and use Similarity to get the top_users
        if distance is None:
            distance = pearson_correlation
        if n_similarity == 'user_similarity':
            self._set_similarity(data_model, n_similarity, distance, nhood_size)
        else:
            raise ValueError('similarity argument must be user_similarity')

        neighborhood = [to_user_id for to_user_id, score in self.similarity[user_id] \
                           if not np.isnan(score) and abs(score) >= minimal_similarity and user_id != to_user_id]
        return neighborhood

class BaseCandidateItemsStrategy(object):
    def candidate_items(self, user_id, data_model, **params):
        raise NotImplementedError("BaseCandidateItemsStrategy is an abstract class.")

class AllPossibleItemsStrategy(BaseCandidateItemsStrategy):
    '''
    Returns all items that have not been rated by the user.
    This strategy is not recommended for large datasets and
    it is the dummiest one.
    '''

    def candidate_items(self, user_id, data_model, **params):
        #Get all the item_ids preferred from the user
        preferences = data_model.items_from_user(user_id)
        #Get all posible items from the data_model
        possible_items = data_model.item_ids()
        return np.setdiff1d(possible_items, preferences, assume_unique=True)


class ItemsNeighborhoodStrategy(BaseCandidateItemsStrategy):
    '''
    Returns all items that have not been rated by the user and were
    preferred by another user that has preferred at least one item that the
    current has preferred too.
    '''

    def candidate_items(self, user_id, data_model, **params):
        #Get all the item_ids preferred from the user
        preferences = data_model.items_from_user(user_id)
        possible_items = np.array([])
        for item_id in preferences:
            item_preferences = data_model.preferences_for_item(item_id)
            if data_model.has_preference_values():
                for user_id, score in item_preferences:
                    possible_items = np.append(possible_items, \
                        data_model.items_from_user(user_id))
            else:
                for user_id in item_preferences:
                    possible_items = np.append(possible_items, \
                        data_model.items_from_user(user_id))
        possible_items = np.unique(possible_items)
        return np.setdiff1d(possible_items, preferences, assume_unique=True)
