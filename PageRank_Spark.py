import pyspark
from pyspark.context import SparkContext
from pyspark import SparkConf

conf = SparkConf()
sc = SparkContext(conf = conf)
sc.setLogLevel("ERROR")

# Load the adjacency list file
AdjList1 = sc.textFile("/home/miles/data/02AdjacencyList.txt")
print(AdjList1.collect())

AdjList2 = AdjList1.map(lambda line : line.split())
AdjList3 = AdjList2.map(lambda x : (x[0],x[1:len(x)]))
AdjList3.persist()
print(AdjList3.collect())

nNumOfNodes = AdjList3.count()
print("Total Number of nodes")
print(nNumOfNodes)

# Initialize each page's rank;
PageRankValues = AdjList3.mapValues(lambda v : 0.2) 
print(PageRankValues.collect())


def FlatMapHelper(x):
    returnList = []
    for val in x[1][0]:
        returnList.append((val,x[1][1]/len(x[1][0])))
    return returnList

# Run 30 iterations
print("Run 30 Iterations")
for i in range(1, 30):
    print("Number of Iterations")
    print(i)
    JoinRDD = AdjList3.join(PageRankValues)
    print("join results")
    print(JoinRDD.collect())
    contributions = JoinRDD.flatMap(FlatMapHelper)
    print("contributions")
    print(contributions.collect())
    accumulations = contributions.reduceByKey(lambda x, y : x + y)
    print("accumulations")
    print(accumulations.collect())
    PageRankValues = accumulations.mapValues(lambda v : .85*v + .03) 
    print("PageRankValues")
    print(PageRankValues.collect())

print("=== Final PageRankValues ===")
print(PageRankValues.collect())

# Write out the final ranks
PageRankValues.coalesce(1).saveAsTextFile("/home/miles/data/PageRankValues_Final3")
