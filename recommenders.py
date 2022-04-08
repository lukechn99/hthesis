import numpy as np
from surprise.prediction_algorithms.predictions import Prediction
import pandas as pd
import time
import math
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.prediction_algorithms.baseline_only import BaselineOnly
from surprise.prediction_algorithms.random_pred import NormalPredictor
from surprise.prediction_algorithms.matrix_factorization import SVD

class ILFM():
    def __init__(self, k=13, inv=0):# we will change k in the next task
      # k: number of hidden features
        self.predict_ratings = None
        self.k=k
        self.inv = inv
        self.bias = 0
        self.user_bias = None
        self.item_bias = None

    def fit(self, trainset, learning_rate=0.001, n_iter=20, gamma = 0.015):
        raw_ratings = pd.DataFrame([x for x in trainset.all_ratings()],
                                   columns=["userId", "movieId", "ratings"])
        self.ratings = raw_ratings.pivot(index='userId', columns='movieId', values='ratings')
        # self.ratings = trainset
        self.ratings.index = self.ratings.index.map(str)# get the list of userId (type str)
        self.ratings.columns = self.ratings.columns.map(str)# get the list of movieId (type str)

        # there are k dataframes with predicted ratings
        self.predict_ratings = None

        ratings = self.ratings.values
        n_users = ratings.shape[0]
        n_movies = ratings.shape[1]
        # here we initialize a random user by feature matrix and a random feature by movie matrix.
        # you will need to optimize the entries of these 2 matrices in your funksvd algorithm
        user_feature = np.random.rand(n_users, self.k) 
        movie_feature = np.random.rand(self.k, n_movies)

        self.user_bias = np.random.rand(n_users)
        self.item_bias = np.random.rand(n_movies)

        # pre-processing: instead of having all ratings, store the exisiting ratings in a set/list
        ratings_map = []
        for i in range(n_users):
          for j in range(n_movies):
            if not np.isnan(ratings[i][j]):
              ratings_map.append((i, j, ratings[i][j]))

        # train ratings matrix
        for epoch in range(n_iter):
          for r in ratings_map:
            # select the right row and column from user_feature and movie_feature matrices
            i = r[0]
            j = r[1]
            row = user_feature[i, :]
            col = movie_feature[:, j]

            # find dot product (predicted_rating)
            predicted_rating = np.dot(row, col) + self.bias + self.user_bias[i] + self.item_bias[j]

            # find the difference between rating and predicted_rating
            diff = r[2] - predicted_rating

            self.bias += learning_rate * (diff - gamma * self.bias)
            self.user_bias[i] += learning_rate * (diff - gamma * self.user_bias[i])
            self.item_bias[j] += learning_rate * (diff - gamma * self.item_bias[j])

            # learn/fit the data for each k
            for feature in range(self.k):
              user_feature[i, feature] += learning_rate * (diff * movie_feature[feature, j] - gamma * user_feature[i, feature])
              movie_feature[feature, j] += learning_rate * (diff * user_feature[i, feature] - gamma * movie_feature[feature, j])
        
        # print(user_feature[0])
        # user_feature[:, self.inv] = abs(user_feature[:, self.inv]-1)
        # user_feature[:, self.inv] *= -1
        # print(user_feature[0])

        ratings = np.matmul(user_feature, movie_feature)
        # add ratings over to self.predicted_ratings
        self.predict_ratings = pd.DataFrame(ratings, index=self.ratings.index, columns=self.ratings.columns)
        user_bias = pd.DataFrame(pd.np.tile(self.user_bias, (n_users, 1)))
        item_bias = pd.DataFrame(pd.np.tile(self.item_bias, (1, n_movies)))
        self.predict_ratings = self.predict_ratings + user_bias + item_bias + self.bias

    def test(self, testset):
      predictions = []
      for userId, movieId, true_rating in testset:
        if userId in self.predict_ratings.index and movieId in self.predict_ratings.columns and not np.isnan(self.predict_ratings.loc[userId, movieId]):
          pred_rating = self.predict_ratings.loc[userId, movieId]
          predictions.append(Prediction(uid=userId, iid=movieId, r_ui=true_rating, est=pred_rating, details={'was_impossible': False}))
        else:
          predictions.append(Prediction(uid=userId, iid=movieId, r_ui=true_rating, est=10, details={'was_impossible': True}))
      return predictions