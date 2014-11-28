#!/usr/bin/env python
from data import load_dataset
from datamodel import MatrixPreferenceDataModel
from similarities import UserSimilarity, ItemSimilarity
from distances import pearson_correlation, euclidean_distances, cosine_distances
from neighborhood import NearestNeighborsStrategy, ItemsNeighborhoodStrategy
from recommender import UserBasedRecommender, ItemBasedRecommender
from evaluator import Evaluator

def start():
    movies = load_dataset()
    model = MatrixPreferenceDataModel(movies['data'])
    option = int(input("Enter: \n 1 for User Based Recommender \n 2 for Item Based Recommender \n"))
    if option != 1 and option != 2:
        print("Invalid Input")
        return
    if option == 1:
        similarity = UserSimilarity(model, cosine_distances)
        neighborhood = NearestNeighborsStrategy()
        recsys = UserBasedRecommender(model, similarity, neighborhood)

    if option == 2:
        similarity = ItemSimilarity(model, cosine_distances)
        neighborhood = ItemsNeighborhoodStrategy()
        recsys = ItemBasedRecommender(model, similarity, neighborhood)

    evaluator = Evaluator()
    all_scores = evaluator.evaluate(recsys, permutation=False)
    print all_scores

if __name__ == '__main__':
    start()
