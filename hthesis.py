from surprise import SVDinv, SVD
from surprise import Dataset, accuracy
from surprise.model_selection import cross_validate, get_cv
import heapq
import matplotlib.pyplot as plt
import numpy as np

# Load the movielens-100k dataset (download it if needed).
# data = Dataset.load_builtin('ml-1m')
data = Dataset.load_builtin('ml-100k')

def run_metrics():
    # Use the famous SVD algorithm.
    # algos = [SVDinv(n_factors=13, inv=0), 
    # SVDinv(n_factors=13, inv=2), 
    # SVDinv(n_factors=13, inv=4), 
    # SVDinv(n_factors=13, inv=6), 
    # SVDinv(n_factors=13, inv=8), 
    # SVDinv(n_factors=13, inv=10), 
    # SVD(n_factors=13)]
    algos = [SVDinv(n_factors=13, scale=0.01), 
    SVDinv(n_factors=13, scale=0.1), 
    SVDinv(n_factors=13, scale=0.5), 
    SVDinv(n_factors=13, scale=1), 
    SVDinv(n_factors=13, scale=5), 
    SVDinv(n_factors=13, scale=10), 
    SVD(n_factors=13)]

    results = []
    # Run 5-fold cross-validation and print results.
    for algo in algos:
        print("new algo")
        results.append(cross_validate(algo, data, measures=['RMSE', 'MAE', 'NOVELTY', 'UNEXPECTEDNESS', 'RELEVANCE', 'MAPE', 'TOPKAVG'], cv=5, verbose=True))
    return results

def run_user(uid):
    # data split
    cv = get_cv(2)
    for (trainset, testset) in cv.split(data):
        # predict for one user to show the difference between SVD and SVDinv
        heap = []
        svd = SVDinv(n_factors=13, inv=0)
        svd.fit(trainset)
        # pred = svd.test(testset)
        # print(accuracy.rmse(pred))
        movies = set([datapoint[1] for datapoint in data.raw_ratings])
        for movie in movies:
            # print(svd.estimate(uid, movie))
            heapq.heappush(heap, (svd.estimate(uid, str(movie)), movie))
        top_k = heapq.nlargest(5, heap)
        # print(heap)

# def graph():
#     results = run_metrics()

#     fig, ax = plt.subplots(2, 3, figsize=(20, 2), sharex=False, sharey=False)
#     fig.subplots_adjust(hspace=0.8, wspace=0.4)
#     X = np.array(["SVDinv@0", "SVDinv@2", "SVDinv@4", "SVDinv@6", "SVDinv@8", "SVDinv@10", "SVD"])
#     x = [0, 1, 2, 3, 4, 5, 6]

#     rmse = results.loc['rmse'].to_numpy()
#     ax[0, 0].bar(X, rmse)
#     ax[0, 0].set_title('RMSE', fontsize=14)
#     ax[0, 0].set_xticks(x)
#     ax[0, 0].set_xticklabels(X, rotation=45)
#     ax[0, 0].set_ylim([0.9, 1.1])

#     mae = [0.7962, 0.7750, 0.7830, 0.7767, 0.7877, 0.7929, 0.7382]
#     ax[1].bar(X, mae)
#     ax[1].set_title('MAE', fontsize=14)
#     ax[1].set_xticks(x)
#     ax[1].set_xticklabels(X, rotation=45)
#     ax[1].set_ylim([0.7, 0.9])

#     novelty = [0.0780, 0.0790, 0.0769, 0.0778, 0.0780, 0.0778, 0.0827]
#     novelty_err = [0.0023, 0.0013, 0.0017, 0.0018, 0.0017, 0.0013, 0.0015]
#     ax[2].bar(X, novelty, yerr=novelty_err)
#     ax[2].set_title('novelty@50', fontsize=14)
#     ax[2].set_xticks(x)
#     ax[2].set_xticklabels(X, rotation=45)
#     ax[2].set_ylim([0.07, 0.09])

#     unexpectedness = [0.3117, 0.3121, 0.3104, 0.3119, 0.3114, 0.3114, 0.3115]
#     unexpectedness_err = [0.0035, 0.0027, 0.0015, 0.0024, 0.0017, 0.0018, 0.0028]
#     ax[3].bar(X, unexpectedness, yerr=unexpectedness_err)
#     ax[3].set_title('unexpectedness', fontsize=14)
#     ax[3].set_xticks(x)
#     ax[3].set_xticklabels(X, rotation=45)
#     ax[3].set_ylim([0.3, 0.32])

run_metrics()