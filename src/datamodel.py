import numpy as np

class MatrixPreferenceDataModel(object):
    '''
    Parameters
    ----------
    dataset dict, shape  = {userID:{itemID:preference, itemID2:preference2},
              userID2:{itemID:preference3,itemID4:preference5}}
    '''
    def __init__(self, dataset):
        self.dataset = dataset
        self.build_model()

    def __getitem__(self, user_id):
        return self.preferences_from_user(user_id)

    def __iter__(self):
        for index, user in enumerate(self.user_ids()):
            yield user, self[user]

    def __len__(self):
        return self.index.shape

    def build_model(self):
        self._user_ids = np.asanyarray(self.dataset.keys())
        self._user_ids.sort()

        self._item_ids = []
        for items in self.dataset.itervalues():
            self._item_ids.extend(items.keys())

        self._item_ids = np.unique(np.array(self._item_ids))
        self._item_ids.sort()

        self.max_pref = -np.inf
        self.min_pref = np.inf
        print 'creating matrix for {0} users and {1} items'.format(self._user_ids.size, self._item_ids.size)

        self.index = np.empty(shape=(self._user_ids.size, self._item_ids.size))
        for userno, user_id in enumerate(self._user_ids):
            for itemno, item_id in enumerate(self._item_ids):
                r = self.dataset[user_id].get(item_id, np.NaN) #Is it to be np.NaN or 0 ?!!
                self.index[userno, itemno] = r

        if self.index.size:
            self.max_pref = np.nanmax(self.index)
            self.min_pref = np.nanmin(self.index)

    def user_ids(self):
        return self._user_ids

    def item_ids(self):
        return self._item_ids

    def preference_values_from_user(self, user_id):
        user_id_loc = np.where(self._user_ids == user_id)
        if not user_id_loc[0].size:
            #user_id not found
            raise UserNotFoundError

        preferences = self.index[user_id_loc]
        return preferences

    def preferences_from_user(self, user_id, order_by_id=True):
        preferences = self.preference_values_from_user(user_id)
        data = zip(self._item_ids, preferences.flatten())
        if order_by_id:
            return [(item_id, preference)  for item_id, preference in data if not np.isnan(preference)]
        else:
            sorted([(item_id, preference)  for item_id, preference in data \
                         if not np.isnan(preference)], key=lambda item: - item[1])

    def has_preference_values(self):
        return True

    def maximum_preference_value(self):
        return self.max_pref

    def minimum_preference_value(self):
        return self.min_pref

    def users_count(self):
        return self._user_ids.size

    def items_count(self):
        return self._item_ids.size

    def items_from_user(self, user_id):
        preferences = self.preferences_from_user(user_id)
        return [key for key, value in preferences]

    def preferences_for_item(self, item_id, order_by_id=True):
        item_id_loc = np.where(self._item_ids == item_id)
        if not item_id_loc[0].size:
            raise ItemNotFoundError('Item not found')
        preferences = self.index[:, item_id_loc]

        data = zip(self._user_ids, preferences.flatten())
        if order_by_id:
            return [(user_id, preference)  for user_id, preference in data \
                         if not np.isnan(preference)]
        else:
            return sorted([(user_id, preference)  for user_id, preference in data \
                         if not np.isnan(preference)], key=lambda user: - user[1])

    def preference_value(self, user_id, item_id):
        item_id_loc = np.where(self._item_ids == item_id)
        user_id_loc = np.where(self._user_ids == user_id)

        if not user_id_loc[0].size:
            raise UserNotFoundError('user_id in the model not found')

        if not item_id_loc[0].size:
            raise ItemNotFoundError('item_id in the model not found')

        return self.index[user_id_loc, item_id_loc].flatten()[0]

class UserNotFoundError(Exception):
    pass
class ItemNotFoundError(Exception):
    pass
