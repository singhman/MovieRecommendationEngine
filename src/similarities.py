import numpy as np

def find_common_elements(source_preferences, target_preferences):
    ''' Returns the preferences from both vectors '''
    src = dict(source_preferences)
    tgt = dict(target_preferences)

    inter = np.intersect1d(src.keys(), tgt.keys())

    common_preferences = zip(*[(src[item], tgt[item]) for item in inter \
            if not np.isnan(src[item]) and not np.isnan(tgt[item])])
    if common_preferences:
        return np.asarray([common_preferences[0]]), np.asarray([common_preferences[1]])
    else:
        return np.asarray([[]]), np.asarray([[]])

###############################################################################
########################## User Similarity ####################################
###############################################################################

class UserSimilarity(object):
    def __init__(self, model, distance, num_best=None):
        self.model = model
        self.distance = distance
        self._set_num_best(num_best)

    def _set_num_best(self, num_best):
        self.num_best = num_best

    def __getitem__(self, source_id):
        all_sims = self.get_similarities(source_id)
        tops = sorted(all_sims, key=lambda x: -abs(x[1]))

        if all_sims:
            item_ids, preferences = zip(*all_sims)
            preferences = np.array(preferences).flatten()
            item_ids = np.array(item_ids).flatten()
            sorted_prefs = np.argsort(-abs(preferences))
            tops = zip(item_ids[sorted_prefs], preferences[sorted_prefs])

        # return at most numBest top 2-tuples (label, sim)
        return tops[:self.num_best] if self.num_best is not None else tops

    def get_similarity(self, source_id, target_id):
        source_preferences = self.model.preferences_from_user(source_id)
        target_preferences = self.model.preferences_from_user(target_id)

        if self.model.has_preference_values():
            source_preferences, target_preferences = \
                find_common_elements(source_preferences, target_preferences)

        if source_preferences.ndim == 1 and target_preferences.ndim == 1:
            source_preferences = np.asarray([source_preferences])
            target_preferences = np.asarray([target_preferences])

        #evaluate the similarity between the two users vectors.
        return self.distance(source_preferences, target_preferences) \
            if not source_preferences.shape[1] == 0 \
                and not target_preferences.shape[1] == 0 else np.array([[np.nan]])

    def get_similarities(self, source_id):
        return [(other_id, self.get_similarity(source_id, other_id))  for other_id, v in self.model]

    def __iter__(self):
        for source_id, preferences in self.model:
            yield source_id, self[source_id]

###############################################################################
########################## Item Similarity ####################################
###############################################################################

class ItemSimilarity(object):
    '''
    Returns the degree of similarity, of two items, based on its preferences by the users.
    Implementations of this class define a notion of similarity between two items.
    Implementations should  return values in the range 0.0 to 1.0, with 1.0 representing
    perfect similarity.
    '''

    def __init__(self, model, distance, num_best=None):
        self.model = model
        self.distance = distance
        self._set_num_best(num_best)

    def _set_num_best(self, num_best):
        self.num_best = num_best

    def get_similarity(self, source_id, target_id):
        source_preferences = self.model.preferences_for_item(source_id)
        target_preferences = self.model.preferences_for_item(target_id)

        if self.model.has_preference_values():
            source_preferences, target_preferences = \
                find_common_elements(source_preferences, target_preferences)

        if source_preferences.ndim == 1 and target_preferences.ndim == 1:
            source_preferences = np.asarray([source_preferences])
            target_preferences = np.asarray([target_preferences])

        #Evaluate the similarity between the two users vectors.
        return self.distance(source_preferences, target_preferences) \
            if not source_preferences.shape[1] == 0 and \
                not target_preferences.shape[1] == 0 else np.array([[np.nan]])

    def get_similarities(self, source_id):
        return [(other_id, self.get_similarity(source_id, other_id)) for other_id in self.model.item_ids()]

    def __iter__(self):
        """
        For each object in model, compute the similarity function against all other objects and yield the result.
        """
        for item_id in self.model.item_ids():
            yield item_id, self[item_id]
