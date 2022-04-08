from surprise.accuracy import mae, rmse
from sklearn.metrics import ndcg_score
from scipy import sparse
from heapq import heappush, nlargest
import math
import pandas as pd
import numpy as np

movie_tags = pd.read_csv('data/movie_tag_rel.csv', index_col=0)

def intra_similarity_aux(items):
    """
    takes a list of items and calculates their intra list similarity with the movie tags from movie_tags
    returns a float
    """
    correlation_sum = 0
    number_of_pairs = 0
    for i in range(len(items)):
      for j in range(i + 1, len(items)):
        correlation_sum += movie_tags.loc[items[i]].corr(movie_tags.loc[items[j]])
        number_of_pairs += 1
    return correlation_sum / max(number_of_pairs, 1)

def intra_similarity(surprise_predictions, k_highest_scores=5):
    """ 
    Calculates the intra-similarity score from surprise predictions
  
    Parameters: 
    surprise_predictions (List of surprise.prediction_algorithms.predictions.Prediction): list of predictions, 
    see https://surprise.readthedocs.io/en/stable/predictions_module.html?highlight=prediction#surprise.prediction_algorithms.predictions.Prediction 
    k_highest_scores (positive integer): Only consider the highest k scores (items) in each user's recommendation list.
  
    Returns: 
    float in [-1, 1.]: The averaged ILS@5 over all users' recommendation lists
    """
    uidset = list(set([p.uid for p in surprise_predictions]))
    iidset = list(set([p.iid for p in surprise_predictions]))
    ratings = pd.DataFrame(index=uidset, columns=iidset)
    
    # we create a hashmap of each user to a top-5 maxheap of (rating, iid)
    users_top_5 = {}
    for user in uidset:
      users_top_5[user] = []

    for prediction in surprise_predictions:
      ratings.loc[prediction.uid, prediction.iid] = prediction.est
      heappush(users_top_5[prediction.uid], (prediction.est, prediction.iid))
    ratings.fillna(0, inplace=True) # now we've transformed the predictions into a userId x movieId pred_rating dataframe    
    # print(ratings.head())
    numerator = 0
    denominator = 0
    empty_user = 0
    for user in uidset:
      top_5 = nlargest(5, users_top_5[user])
      # discard the rating, only look at the movie ID
      top_5_and_valid = [r[1] for r in top_5 if r[1] in movie_tags.index.values]
      if len(top_5_and_valid) > 1:
        similarity_for_user = intra_similarity_aux(top_5_and_valid)
        numerator += similarity_for_user
        denominator += 1
        print("numerator", numerator, "denominator", denominator)
      else:
        empty_user += 1
    # print("there are {}/{} users with empty lists".format(empty_user, denominator))
    intra_similarity_score = numerator / (2 * max(denominator, 1))
    # print("IS: {}".format(intra_similarity_score))
    return intra_similarity_score

def vargasNovelty(surprise_predictions, k=50):
    '''
    Similar to how we take nDCG@k, where k is some value like 5, vargasNovelty
    is to be taken by calculating novelty for each inverted prediction matrix 
    based on the top k values
    surprise_predictions is a list of lists. Each outer list represents a prediction
    matrix 
    '''
    total_nov = []
    uidset = list(set([p.uid for p in surprise_predictions]))
    iidset = list(set([p.iid for p in surprise_predictions]))
    ratings = pd.DataFrame(index=uidset, columns=iidset)
    for prediction in surprise_predictions:
      ratings.loc[prediction.uid, prediction.iid] = prediction.est
    ratings.fillna(0, inplace=True)
    # find top 5 from each "predictions"
    users_top_5 = {}
    for user in uidset:
      users_top_5[user] = []

    for p in surprise_predictions:
      heappush(users_top_5[p.uid], (p.est, p.iid))
    # calculate novelty
    for user in uidset:
      top_5 = nlargest(5, users_top_5[user])
      for i in range(len(top_5)):
        ith_item = top_5[i][1]
        ith_item_score = top_5[i][0]
        # this is a slightly modified novelty function that removes the constant and simplifies rel() and adjusted log() for zero indexing
        nov = (1/max(1, math.log(i+1,2))) * 2**(float(ith_item_score)-5) * (1-np.count_nonzero(ratings.loc[:, ith_item])/len(uidset))
      total_nov.append(nov)
    avg_nov = np.mean(total_nov)
    return avg_nov

# def basicNovelty(surprise_predictions, k_highest_scores=5):
#     # create the predicted ratings matrix and fill 0 for empty spaces
#     uidset = list(set([p.uid for p in surprise_predictions]))
#     iidset = list(set([p.iid for p in surprise_predictions]))
#     ratings = pd.DataFrame(index=uidset, columns=iidset)
#     for prediction in surprise_predictions:
#       ratings.loc[prediction.uid, prediction.iid] = prediction.est
#     ratings.fillna(0, inplace=True)

#     # using the rating_df, defined as "rating_df = tmp.pivot(index='userId', columns='movieId', values='rating')", check for novelty

#     # filter for top results by user
#     users_top_5 = {}
#     for user in uidset:
#       users_top_5[user] = []

#     for p in surprise_predictions:
#       heappush(users_top_5[p.uid], (p.est, p.iid, p.r_ui))
    
#     seen = 0
#     unseen = 0

#     # we find the ratio of seen to unseen 
#     for user in users_top_5.keys():
#       top_5 = nlargest(5, users_top_5[user])
#       for item in top_5:
#         # print(rating_df.loc[user, item[1]])
#         if np.isnan(rating_df.loc[user, item[1]]):
#           unseen += 1
#         else:
#           seen += 1
#     return seen / max(unseen + seen, 1)

def unexpectedness_aux(user_id, top_5, ratings):
  '''
  roughly, we're following the equation -log( p(i,j)/p(i)p(j) ) / log(p(i,j))
  The p(i) is going to be calculated for each item in the user's item corpus
  calculated ahead of time in the pre-processing step

  user_id: userId
  top_5: List[Tuple(est, itemId)]
  ratings: pd.df
  '''
  total_pmi = []

  # pre-process the stats for items the user has rated
  user_rated_items = np.array(ratings.index.to_list())[ratings[user_id] > 0]

  # item-to-p(i) map
  p_i = {}

  # item-to-p(i) indices map. this is used to calculate p(i,j) later
  p_ii = {}

  print(user_rated_items)
  for item_id in user_rated_items:
    print("item_id", item_id)

    # probability of item i considering all items
    p_i[item_id] = np.count_nonzero(ratings.loc[:,item_id])/ratings.shape[1]
    p_ii[item_id] = np.nonzero(np.array(ratings.loc[:,item_id]))

  for item in top_5:
    item_id = item[1]   # int
    p_j = np.count_nonzero(ratings.loc[:,item_id])/ratings.shape[1]   # float

    # compare this item with every item that the user has rated
    for comp_item_id in user_rated_items:
      print("comp_item_id", comp_item_id)

      # optimize with this later: https://stackoverflow.com/questions/63317109/python-get-column-name-by-non-zero-value-in-row
      # extract item_ids that have nonzero ratings for this user
      jitems = []
      for r, name in zip(ratings.loc[:,item_id], ratings.index):
        if r != 0:
          jitems.append(int(name))

      # p_ij = len(set(p_ii[item_id2][0].tolist() + ratings.loc[:,item_id].values.flatten().tolist()))/ratings.shape[1]
      p_ij = len(set(p_ii[comp_item_id][0].tolist()) & set(jitems))/ratings.shape[1]
      print(p_ij)
      print(p_i[comp_item_id] * p_j)
      print("top", -math.log(max(p_ij / max((p_i[comp_item_id] * p_j), 1), 1), 2))
      print("bottom", math.log(max(p_ij, 1), 2))
      total_pmi.append(-math.log(max(p_ij / max((p_i[comp_item_id] * p_j), 1), 1), 2) / math.log(max(p_ij, 1.1), 2))

  avg_pmi = np.mean(total_pmi)
  return avg_pmi

def unexpectedness(surprise_predictions):
  '''
  We will measure unexpectedness of the top 5 recommendations. 
  To do this, we will take each of the recommendations, and calculate its unexpectedness
  when observing the body of items rated by the user
  More specifically, we are taking the user's item corpus I = {i1, i2, ..., in}
  '''
  # unexpectedness will be a collection of the unexpectedness score for each user with repeats of users for ilfm
  total_unexpectedness = []

  return 0
  # find top five for each user
  uidset = list(set([p.uid for p in surprise_predictions]))
  iidset = list(set([p.iid for p in surprise_predictions]))
  ratings = pd.DataFrame(index=uidset, columns=iidset)
  for prediction in surprise_predictions:
    ratings.loc[prediction.uid, prediction.iid] = prediction.est
  ratings.fillna(0, inplace=True)

  users_top_5 = {}
  # initialize a heap for each user
  for user in uidset:
    users_top_5[user] = []

  # push all predictions onto user heaps
  for prediction in surprise_predictions:
    heappush(users_top_5[prediction.uid], (prediction.est, prediction.iid))

  # calculate the unexpectedness for each user
  for user in uidset:
    total_unexpectedness.append(unexpectedness_aux(user, nlargest(5, users_top_5[user]), ratings))

  avg_unexpectedness = np.mean(total_unexpectedness)
  return avg_unexpectedness


def relevance(surprise_predictions):
    return 0

def ndcg(surprise_predictions, k=5):
    total_ndcg = []
    uidset = list(set([p.uid for p in surprise_predictions]))
    iidset = list(set([p.iid for p in surprise_predictions]))

    est = pd.DataFrame(0, index=iidset, columns=uidset)
    true = pd.DataFrame(0, index=iidset, columns=uidset)

    for prediction in surprise_predictions:
      est.loc[prediction.iid, prediction.uid] = prediction.est
      true.loc[prediction.iid, prediction.uid] = prediction.r_ui

    total_ndcg.append(ndcg_score(true, est, k=k))
    # print("NDCG: {}".format(ndcg_result))
    avg_ndcg = np.mean(total_ndcg)
    return avg_ndcg