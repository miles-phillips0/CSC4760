import pyspark
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import sklearn as sk
from sklearn import datasets
from sklearn.datasets import load_svmlight_file
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('webagg')

conf = SparkConf()
sc = SparkContext(conf = conf)
sc.setLogLevel("ERROR")
sqlContext = pyspark.SQLContext(sc)

# Loads data.
dataset = sqlContext.read.format("libsvm").load("/home/miles/data/kmeans_input.txt")

# Trains a k-means model.
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dataset)

# Make predictions
predictions = model.transform(dataset)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

ScatterPlotX = dataset.select(dataset.columns[1]).rdd.map(lambda x : x[0][0]).collect()
ScatterPlotY = dataset.select(dataset.columns[1]).rdd.map(lambda x : x[0][1]).collect()
ScatterPlotLabel = dataset.select(dataset.columns[0]).rdd.map(lambda x : x[0]).collect()

plt.scatter(ScatterPlotX,ScatterPlotY,c=ScatterPlotLabel)
plt.show()

