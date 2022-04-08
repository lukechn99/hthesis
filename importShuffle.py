from surprise.model_selection import KFold
# from sklearn.model_selection import KFold
from surprise import Dataset, accuracy
from surprise.reader import Reader
import random
import copy

data = Dataset.load_builtin('ml-100k')
tmp = pd.DataFrame(data.raw_ratings, columns=["userId", "movieId", "rating", "tstamp"])
rating_df = tmp.pivot(index='userId', columns='movieId', values='rating')

def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

non_rated_movies = []
for userId in rating_df.index:
  tmp = rating_df.loc[userId].loc[rating_df.loc[userId].isna()].index
  for movieId in tmp:
    non_rated_movies.append((userId, movieId, 0))

n_splits=5
non_rated_testset = partition(non_rated_movies, n_splits)

def my_split(data, n_splits=n_splits, random_state=2021):
    kf = KFold(n_splits=n_splits, random_state=random_state)

    dataset = []
    i = 0
    tmp = copy.deepcopy(data)
    for trainset, testset_rated in kf.split(tmp):
      testset_withunrated = copy.deepcopy(testset_rated)
      testset_withunrated.extend(non_rated_testset[i][:len(testset_rated)])
      dataset.append((trainset, testset_rated, testset_withunrated))
      i += 1
    return dataset