# Unsupervised Learning Final

##High Level Overview: Through this analysis, I pulled in data from Kaggle, cleaned and split the data and ran through a K Means Clustering model for further Analysis. Below, I discuss the insights and lessons learned. 

##Read in all the packages
import pandas as pd
import numpy as np
import seaborn as sns
import umap
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from yellowbrick.cluster import KElbowVisualizer
from sklearn.manifold import TSNE

#Data Preparation
    #For this data set, I pulled pulled in a dataset from Kaggle
originalData = pd.read_csv('/kaggle/input/tabular-playground-series-jul-2022/data.csv')
data_types = originalData.dtypes
categorical_features = (data_types[data_types == 'int64'].index).drop('id')
numerical_features = data_types[data_types == 'float64'].index
data = originalData.drop("id", axis=1)
print("Number of nulls: ", data.isna().sum().sum())

data.head().T

data.describe().T

sns.heatmap(originalData.corr(), cmap='YlOrRd')

scaler = StandardScaler()
encoder = OneHotEncoder()
data[numerical_features] = scaler.fit_transform(data[numerical_features])
encoded_data = encoder.fit_transform(data[categorical_features])

#Reordering the columns

reorder = list(numerical_features) + list(categorical_features)
data = data[reorder]

#Feature Selection below using PCA
scores = []
for i in range(1, 30):
    testData = data.copy()
    decomposer = PCA(n_components=i)
    decomposer.fit_transform(testData)
    scores.append(sum(decomposer.explained_variance_ratio_))
print(scores)
plt.plot(scores)


NO_OF_COMPONENTS = 6
names = ["Principal Component {}".format(i) for i in range(1, NO_OF_COMPONENTS+1)]
decomposer = PCA(n_components=NO_OF_COMPONENTS)
dataPCA = decomposer.fit_transform(data)

#Explained variance 
explainedVariance = sum(decomposer.explained_variance_ratio_)
dataPCA = pd.DataFrame(dataPCA, columns=names)
print("Explained variance: ", explainedVariance*100, "%") ##82%


#Hyper Parameter Selection

##After further digging, we can conclude k=7 works best
newData = data.iloc[:, 0:-1].to_numpy()
newDataPCA = dataPCA.iloc[:, 0:-1].to_numpy()

model = KMeans()
visualizer = KElbowVisualizer(model, k=(2, 20), timings=True)
visualizer.fit(newDataPCA)
visualizer.show()

#Set up for success
k_range = range(2, 10)

silhouetteScores = []
DBindexScores = []

for k in tqdm(k_range):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(newDataPCA)
    score1 = silhouette_score(newDataPCA, labels, metric='euclidean')
    score2 = davies_bouldin_score(newDataPCA, labels)
    silhouetteScores.append((k, score1))
    DBindexScores.append((k, score2))

ax_x, ax_y = zip(*silhouetteScores)
fig = plt.figure(figsize=(10,5))
ax = plt.axes()
plt.xticks(k_range)
plt.grid()
plt.plot(ax_x, ax_y)

ax_x, ax_y = zip(*DBindexScores)
fig = plt.figure(figsize=(10,5))
ax = plt.axes()
plt.xticks(k_range)
plt.grid()
plt.plot(ax_x, ax_y)



# KMeans and Clusters
MODELKMEANS = KMeans(n_clusters=7)
CLUSTERS = MODELKMEANS.fit_predict(dataPCA)
originalData["Predicted"] = CLUSTERS
result = originalData.filter(["id", "Predicted"])
result.to_csv("/kaggle/working/KmeansResult.csv", index=False)


labels = MODELKMEANS.fit_predict(dataPCA)
dataPCA['Cluster'] = labels
sns.scatterplot(x="Principal Component 1", y="Principal Component 2", hue = 'Cluster',  data=dataPCA, palette='viridis')

fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.scatter(dataPCA["Principal Component 1"], dataPCA["Principal Component 2"], dataPCA["Principal Component 3"], c = dataPCA['Cluster'], cmap='viridis')


tsne = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=1000, verbose=1)
tsneData = dataPCA.copy()
tsne_results = tsne.fit_transform(tsneData)
tsneData['tsne-2d-one'] = tsne_results[:,0]
tsneData['tsne-2d-two'] = tsne_results[:,1]
sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue = 'Cluster',  data=tsneData, palette='viridis')


reducer = umap.UMAP(n_neighbors=50, min_dist=0.3, n_components=2)
umapData = dataPCA.copy()
embedding = reducer.fit_transform(umapData)
umapData['umap-2d-one'] = embedding[:,0]
umapData['umap-2d-two'] = embedding[:,1]
sns.scatterplot(x="umap-2d-one", y="umap-2d-two", hue = 'Cluster',  data=umapData, palette='viridis')
#Cluster vs Cosine Similarity

#Heatmap below
sns.heatmap(cosine_similarity(MODELKMEANS.cluster_centers_), cmap='YlOrRd')

k = 5
nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
distances, indices = nbrs.kneighbors(data)

k_distances = np.sort(distances[:, k], axis=0)

plt.plot(np.arange(len(data)), k_distances)
plt.xlabel('Samples')
plt.ylabel(f'{k}-Distance')
plt.title(f'K-Distance Graph (k={k})')
plt.show()



MODELDBSCAN = DBSCAN(eps=2, min_samples=5)
MODELDBSCAN.fit(dataPCA)
clusters = MODELDBSCAN.labels_


test = umapData.copy()
test['Cluster'] = clusters
sns.scatterplot(x="umap-2d-one", y="umap-2d-two", hue = 'Cluster',  data=test, palette='viridis')

#Insights: Through this excersise, I once again saw the importance of choosing a good k value and thuroughly validating and testing the data and the models. 