import heapq
import matplotlib.pyplot as plt
from importShuffle import *
from evaluation import *
from recommenders import SVD

def findBestK(data, min_k, max_k, folds=5):
    split = my_split(data)
    results = pd.DataFrame(0, index=["MAE", "RMSE"], columns=[i for i in range(min_k, max_k + 1)])

    for k in range(min_k, max_k + 1):
        for fold in split:
            # extract datasets
            trainset = fold[0]
            testset_rated = fold[1]
            testset_withunrated = fold[2]

            svd = SVD(n_factors=k)
            svd.fit(trainset)
            prediction = svd.test(testset_rated)

            # record the accuracy
            results.loc["MAE", k] += accuracy.mae(prediction)
            results.loc["RMSE", k] += accuracy.rmse(prediction)
    results = results / folds
    return results

results = findBestK(data, 0, 100)

results.loc["RMSE"].plot.bar()
plt.ylim(0.93, 0.945)
plt.figure(figsize=(20,100))

print(results)