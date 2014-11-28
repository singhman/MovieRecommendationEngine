import numpy as np
from base import BaseEstimator
from neighborhood import NearestNeighborsStrategy, ItemsNeighborhoodStrategy
from similarities import find_common_elements

############################################################################################
############################### USER BASED RECOMMENDER #####################################
############################################################################################

class UserBasedRecommender(BaseEstimator):
    """
    User Based Collaborative Filtering Recommender.

    Attributes
    -----------
    `model`: The data model instance that will be data source
         for the recommender.

    `similarity`: The User Similarity instance that will be used to
        score the users that are the most similar to the user.

    `neighborhood_strategy`: The user neighborhood strategy that you
         can choose for selecting the most similar users to find
         the items to recommend.
         default = NearestNeighborsStrategy

    `capper`: bool (default=True)
        Cap the preferences with maximum and minimum preferences
        in the model.
    `with_preference`: bool (default=False)
        Return the recommendations with the estimated preferences if True.
    """

    def __init__(self, model, similarity, neighborhood_strategy=None,
                capper=True, with_preference=False):
        self.model = model
        self.with_preference = with_preference
        self.similarity = similarity
        self.capper = capper
        if neighborhood_strategy is None:
            self.neighborhood_strategy = NearestNeighborsStrategy()
        else:
            self.neighborhood_strategy = neighborhood_strategy

    def recommend(self, user_id, how_many=None, **params):
        self.set_params(**params)
        candidate_items = self.all_other_items(user_id, **params)
        recommendable_items = self._top_matches(user_id, candidate_items, how_many)
        return recommendable_items

    def all_other_items(self, user_id, **params):
        n_similarity = params.get('n_similarity', 'user_similarity')
        distance = params.get('distance', self.similarity.distance)
        nhood_size = params.get('nhood_size', None)

        nearest_neighbors = self.neighborhood_strategy.user_neighborhood(user_id, self.model, n_similarity, distance, nhood_size, **params)
        items_from_user_id = self.model.items_from_user(user_id)

        possible_items = []
        for to_user_id in nearest_neighbors:
            possible_items.extend(self.model.items_from_user(to_user_id))
        possible_items = np.unique(np.array(possible_items).flatten())

        return np.setdiff1d(possible_items, items_from_user_id)

    def estimate_preference(self, user_id, item_id, nearest_neighbors, similarities, user_pref_mean, neighbors_mean, **params):
        preference = self.model.preference_value(user_id, item_id)
        if not np.isnan(preference):
            return preference

        preference = 0.0
        total_similarity = 0.0

        prefs = np.array([self.model.preference_value(to_user_id, item_id)
                 for to_user_id in nearest_neighbors])

        similarities = similarities[~np.isnan(prefs)]
        neighbors_mean = neighbors_mean[~np.isnan(prefs)]
        prefs = prefs[~np.isnan(prefs)]

        #print 'preferences = {0}\nneighbors_means = {1}\nsimilarities = {2}'.format(prefs,neighbors_mean, similarities)
        prefs_sim = np.sum((prefs[~np.isnan(similarities)] - neighbors_mean[~np.isnan(similarities)])*
                             similarities[~np.isnan(similarities)])
        total_similarity = np.sum(abs(similarities[~np.isnan(similarities)]))
        if total_similarity == 0.0 or not similarities[~np.isnan(similarities)].size:
            return np.nan

        estimated = user_pref_mean + (prefs_sim / total_similarity)
        #print 'Estimated Preference is {0} for item {1}\n'.format(estimated, item_id)
        if self.capper:
            max_p = self.model.maximum_preference_value()
            min_p = self.model.minimum_preference_value()
            estimated = max_p if estimated > max_p else min_p \
                     if estimated < min_p else estimated
        return estimated

    def curry_estimate_preference(self, nearest_neighbors, similarities, user_pref_mean, neighbors_mean):
        '''
        Currying nearest neighbors of user for optimization
        '''
        def estimate_preference_curried(user_id, item_id):
            return self.estimate_preference(user_id, item_id, nearest_neighbors, similarities, user_pref_mean, neighbors_mean)
        return estimate_preference_curried

    def _top_matches(self, source_id, target_ids, how_many=None, **params):
        if target_ids.size == 0:
            return np.array([])

        nearest_neighbors, similarities, user_pref_mean, neighbors_mean = self.user_sim_neighbors(source_id, **params)
        estimate_preferences = np.vectorize(self.curry_estimate_preference(nearest_neighbors, similarities, user_pref_mean, neighbors_mean))
        preferences = estimate_preferences(source_id, target_ids)
        preference_values = preferences[~np.isnan(preferences)]
        target_ids = target_ids[~np.isnan(preferences)]
        sorted_preferences = np.lexsort((preference_values,))[::-1]
        sorted_preferences = sorted_preferences[0:how_many] \
             if how_many and sorted_preferences.size > how_many \
                else sorted_preferences

        if self.with_preference:
            top_n_recs = [(target_ids[ind], preferences[ind]) for ind in sorted_preferences]
        else:
            top_n_recs = [target_ids[ind] for ind in sorted_preferences]
        return top_n_recs

    def user_sim_neighbors(self, user_id, **params):
        n_similarity = params.pop('n_similarity', 'user_similarity')
        distance = params.pop('distance', self.similarity.distance)
        nhood_size = params.pop('nhood_size', None)
        nearest_neighbors = self.neighborhood_strategy.user_neighborhood(user_id, self.model, n_similarity, distance, nhood_size, **params)

        source_preferences = self.model.preferences_from_user(user_id)
        neighbors_mean = []
        for neighbor in nearest_neighbors:
            neighbor_preferences = self.model.preferences_from_user(neighbor)
            if self.model.has_preference_values():
                _, neighbor_preferences = find_common_elements(source_preferences, neighbor_preferences)
                if not neighbor_preferences.shape[1] == 0:
                    neighbors_mean.append(np.mean(np.array(neighbor_preferences)))
                else:
                    neighbors_mean.append(np.nan)

        neighbors_mean = np.array(neighbors_mean)
        _, source_preferences = zip(*source_preferences)
        user_preferences = np.array(source_preferences)
        user_pref_mean = np.mean(user_preferences[~np.isnan(user_preferences)])
        similarities = np.array([self.similarity.get_similarity(user_id, to_user_id) for to_user_id in nearest_neighbors]).flatten()
        return nearest_neighbors, similarities, user_pref_mean, neighbors_mean

    def _estimate_preference(self, user_id, item_id, **params):
        preference = self.model.preference_value(user_id, item_id)
        if not np.isnan(preference):
            return preference

        n_similarity = params.pop('n_similarity', 'user_similarity')
        distance = params.pop('distance', self.similarity.distance)
        nhood_size = params.pop('nhood_size', None)

        nearest_neighbors = self.neighborhood_strategy.user_neighborhood(user_id,
                self.model, n_similarity, distance, nhood_size, **params)

        preference = 0.0
        total_similarity = 0.0

        similarities = np.array([self.similarity.get_similarity(user_id, to_user_id)
                for to_user_id in nearest_neighbors]).flatten()

        prefs = np.array([self.model.preference_value(to_user_id, item_id)
                 for to_user_id in nearest_neighbors])

        prefs = prefs[~np.isnan(prefs)]
        similarities = similarities[~np.isnan(prefs)]

        prefs_sim = np.sum(prefs[~np.isnan(similarities)] *
                             similarities[~np.isnan(similarities)])
        total_similarity = np.sum(similarities)

        if total_similarity == 0.0 or \
           not similarities[~np.isnan(similarities)].size:
            return np.nan

        estimated = prefs_sim / total_similarity

        if self.capper:
            max_p = self.model.maximum_preference_value()
            min_p = self.model.minimum_preference_value()
            estimated = max_p if estimated > max_p else min_p \
                     if estimated < min_p else estimated
        print 'Estimated preference is {0} for item {1}'.format(estimated,item_id)
        return estimated

############################################################################################
############################### ITEM BASED RECOMMENDER #####################################
############################################################################################

class ItemBasedRecommender(BaseEstimator):
    """
    Item Based Collaborative Filtering Recommender.

    Parameters
    -----------
    data_model: The data model instance that will be data source
         for the recommender.

    similarity: The Item Similarity instance that will be used to
        score the items that will be recommended.

    items_selection_strategy: The item candidates strategy that you
     can choose for selecting the possible items to recommend.
     default = ItemsNeighborhoodStrategy

    capper: bool (default=True)
        Cap the preferences with maximum and minimum preferences
        in the model.
    with_preference: bool (default=False)
        Return the recommendations with the estimated preferences if True.
    """

    def __init__(self, model, similarity, items_selection_strategy=None,
                capper=True, with_preference=False):
        self.model = model
        self.with_preference = with_preference
        self.similarity = similarity
        self.capper = capper
        if items_selection_strategy is None:
            self.items_selection_strategy = ItemsNeighborhoodStrategy()
        else:
            self.items_selection_strategy = items_selection_strategy

    def recommend(self, user_id, how_many=None, **params):
        self.set_params(**params)
        candidate_items = self.all_other_items(user_id)
        recommendable_items = self._top_matches(user_id, candidate_items, how_many)

        return recommendable_items

    def estimate_preference(self, user_id, item_id, **params):
        '''
        Returns
        -------
        Return an estimated preference if the user has not expressed a
        preference for the item, or else the user's actual preference for the
        item. If a preference cannot be estimated, returns None.
        '''
        preference = self.model.preference_value(user_id, item_id)

        if not np.isnan(preference):
            return preference

        prefs = self.model.preferences_from_user(user_id)

        if not self.model.has_preference_values():
            prefs = [(pref, 1.0) for pref in prefs]

        similarities = \
            np.array([self.similarity.get_similarity(item_id, to_item_id) \
            for to_item_id, pref in prefs if to_item_id != item_id]).flatten()

        prefs = np.array([pref for it, pref in prefs])
        prefs_sim = np.sum(prefs[~np.isnan(similarities)] *
                             similarities[~np.isnan(similarities)])
        total_similarity = np.sum(similarities)

        if total_similarity == 0.0 or \
           not similarities[~np.isnan(similarities)].size:
            return np.nan

        estimated = prefs_sim / total_similarity

        if self.capper:
            max_p = self.model.maximum_preference_value()
            min_p = self.model.minimum_preference_value()
            estimated = max_p if estimated > max_p else min_p \
                     if estimated < min_p else estimated
        return estimated

    def all_other_items(self, user_id, **params):
        return self.items_selection_strategy.candidate_items(user_id, \
                            self.model)

    def _top_matches(self, source_id, target_ids, how_many=None, **params):
        if target_ids.size == 0:
            return np.array([])

        estimate_preferences = np.vectorize(self.estimate_preference)
        preferences = estimate_preferences(source_id, target_ids)

        preference_values = preferences[~np.isnan(preferences)]
        target_ids = target_ids[~np.isnan(preferences)]

        sorted_preferences = np.lexsort((preference_values,))[::-1]

        sorted_preferences = sorted_preferences[0:how_many] \
             if how_many and sorted_preferences.size > how_many \
                else sorted_preferences

        if self.with_preference:
            top_n_recs = [(target_ids[ind], \
                     preferences[ind]) for ind in sorted_preferences]
        else:
            top_n_recs = [target_ids[ind]
                 for ind in sorted_preferences]

        return top_n_recs
