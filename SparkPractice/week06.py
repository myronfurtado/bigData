#install Packages if needed(copy paste in VS code terminal)
#!pip install pyspark

#!pip install jupyter
#!pip install pip
#!pip install pandas
#!pip install numpy
#!pip install matplotlib
#!pip install matplotlib-inline
#!pip install seaborn

import numpy
import pandas
import matplotlib

#my code was not working without this. had to use findspark and the use findspark to find pyspark
import findspark
findspark.init()

#import pyspark library and SparkSession class
import pyspark
from pyspark.sql import SparkSession
#create the SparkSession
spark = SparkSession.builder.getOrCreate()
textdf = spark.sql("select 'spark' as hello")
textdf.show()

#for testing if spark works (with graph)
#import matplotlib.pyplot as plt
#plt.plot([1, 2, 3, 4])
#plt.ylabel('some numbers')
#plt.show()

#
sentenceDataFrame = spark.createDataFrame([
    (0, "Hi I heard about Spark"),
    (1, "I wish Java could use case classes"),
    (2, "Logistic,regression,models,are,neat")
    ], ["id", "sentence"])
#sentenceDataFrame.show()
#convert spark df to pandas df
pandasDF = sentenceDataFrame.toPandas()
#print pandas df (use print statement for pandas dataframes)
print(pandasDF)