import pandas as pd
import warnings
from evaluation import *
from importShuffle import *
from recommenders import *
warnings.filterwarnings('ignore')

# algs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'knn_model', 'baseline_model', 'random_model', 'svd']
# algs = ['0', '3', '6', '9', '12']
algs = ["0"]
# evaluators to be used
measurement_names = ['rmse','mae','ndcg','intra_similarity','novelty','unexpectedness','relevance']

evaluators = [rmse, mae, ndcg, intra_similarity, vargasNovelty, unexpectedness, relevance]

metric_report = pd.DataFrame(0, index=algs, columns=measurement_names)
latent_features = 13
folds = 5

# suppress pandas dataframe truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# load data from drive
# without using a Dataset object
# r_cols = ['userId', 'movieId', 'rating', 'tstamp']
# udata = pd.read_csv('drive/MyDrive/Colab/ml-100k/u.data', sep='\t', names=r_cols)
# total_split = my_split(udata)

# load data from drive with Reader and Dataset
# reader = Reader(line_format=u'user item rating', sep='\t', rating_scale=(1, 5), skip_lines=1)
# data = Dataset.load_from_file('drive/MyDrive/Colab/ml-100k/u.data', reader=reader)
# total_split = my_split(data)

# load data from builtin
data = Dataset.load_builtin('ml-100k')
total_split = my_split(data)

for i in range(folds):
  trainset, testset_rated, testset_with_unrated = total_split[i]

  # instantiate models that correspond to "algs"
  models = []
  for l in range(0, latent_features, 13):
    models.append(ILFM(latent_features, i))
  knn_model = KNNBasic()
  baseline_model = BaselineOnly()
  random_model = NormalPredictor()
  svd = SVD(n_factors=latent_features)
  # models = [ilfm_model, ilfm_model2, ilfm_model3, svd, random_model, baseline_model, knn_model]
  # models += [knn_model, baseline_model, random_model, svd]
  
  # train datasets
  for name, model in zip(algs, models):
    # print("training", name)
    model.fit(trainset)

  # predict ratings
  predictions = []
  for name, model in zip(algs, models):
    # print("predicting for", name)
    predictions.append(model.test(testset_rated))
                
  # record accuracy
  for name, pred in zip(algs, predictions):
    for metric, evaluate in zip(measurement_names, evaluators):
      score = evaluate(pred)
      print(metric, score)
      metric_report.loc[name, metric] += score

metric_report = metric_report / folds
print(metric_report)

# graph
import matplotlib.pyplot as plt

# plt.figure(figsize=(20,10))

ilfm_metric = metric_report[:13]
other_metric = metric_report[13:]

# ilfm_metric["novelty"].plot.bar()
# plt.ylim(0.19, 0.21)

# ilfm_metric["rmse"].plot.bar()
# plt.ylim(1, 1.5)

# ilfm_metric["mae"].plot()
# plt.ylim(1, 1.5)

# metric_report["novelty"].plot.bar()
# plt.ylim(0.15, 0.25)

metric_report["rmse"].plot.bar()
plt.ylim(0.5, 2)

# plt.figure(figsize=(20,10))
metric_report["novelty"].plot.bar()
plt.ylim(0.15, 0.25)