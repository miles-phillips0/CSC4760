import pyspark
from pyspark.context import SparkContext
from pyspark import SparkConf

#initialize spark and sql contexts
conf = SparkConf()
sc = SparkContext(conf = conf)
sc.setLogLevel("ERROR")
sqlContext = pyspark.SQLContext(sc)


#load all files as dataframes
cityStateMap = sqlContext.read.json("/home/miles/data/cityStateMap.json")
cityStateMap.show()
tweets = sqlContext.read.json("/home/miles/data/tweets.json")
tweets.show()


#replace each city in tweets with its corresponding state
for row in cityStateMap.toLocalIterator():
    tweets = tweets.replace(row[0],row[1])
tweets.show()

#drop the user column
tweets = tweets.drop('user')
tweets.show()

#convert the date frame into an RDD
tweetsRDD = tweets.rdd
print(tweetsRDD.collect())

#map the values of the RDD to 1
tweetsRDD2 = tweetsRDD.mapValues(lambda v : 1)
print(tweetsRDD2.collect())

#reduce by key to aggregate the tweets
tweetsRDD3 = tweetsRDD2.reduceByKey(lambda x, y : x+y)
print(tweetsRDD3.collect())

#convert RDD back into a dataframe
outputColumns = ["state","count"]
output = tweetsRDD3.toDF(outputColumns)
output.show()