#import python packages that are needed
from operator import index
from unicodedata import numeric
from matplotlib import axis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#To remove warning from the terminal when doing the box plots
from os import environ
def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"
if __name__ == "__main__":
    suppress_qt_warnings()
#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#my code was not working without this. had to use findspark and the use findspark to find pyspark
#import findspark
#findspark.init()

#import pyspark library and SparkSession class
import pyspark
from pyspark.sql import SparkSession
#create the SparkSession
spark = SparkSession.builder.getOrCreate()
textdf = spark.sql("select 'spark' as hello")
textdf.show()

#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#loading the dataset into a Pyspark dataframe using Python
df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)

#n=number of rows to display, truncate= no of characters
##df.show(n=20, truncate=50)

#Printing the datataset Schema
df.printSchema()

#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#task 2

#convert spark df to pandas df
pandas_df = df.toPandas()
#print(pandas_df)

#Selects the only the rows where the status is "Normal" and saves them to a variable dfNormal
dfNormal = pandas_df.loc[pandas_df['Status'] == "Normal"]
#print(dfNormal)
#Selects the only the rows where the status is "Abnormal" and saves them to a variable dfAbnormal
dfAbnormal = pandas_df.loc[pandas_df['Status'] == "Abnormal"]
#print(dfAbnormal)


