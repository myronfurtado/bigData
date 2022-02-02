

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
#df.printSchema()

rows= str(df.count())
##print("The datset is made of: "+ rows + " rows \n")
#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#task 1

# Find count for empty, None, Null, Nan with string literals.
from pyspark.sql.functions import col, isnan, when, count

df1 = df.select([count(when(col(c).contains('None') | \
                            col(c).contains('NONE') | \
                            col(c).contains('NULL') | \
                            col(c).contains('null') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in df.columns])
df1.show()
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

#function to calculate the Minimum for features of dfNormal or dfAbnormal groups
def minValues(dfType):
    #selects the Minimum values for all columns of only rows with Normal features
    dfValueMin = dfType.min()
    #drop the Status column for better presentation
    dfValueMin = dfValueMin.drop(['Status'])
    return dfValueMin
#Used to print the value of the return statement.    print(functionName(dfNormal or dfAbnormal))
#Print the minimum value for Normal group
##print("\nThe minimum values for Normal group")
##print(minValues(dfNormal))
#Print the minimum value for Abnormal group
##print("\nThe minimum values for Abnormal group")
##print(minValues(dfAbnormal))

#function to calculate the Maximum for features of dfNormal or dfAbnormal groups
def maxValues(dfType):
    #selects the Maximum values for all columns of only rows with Normal features
    dfValueMax = dfType.max()
    #drop the Status column for better presentation
    dfValueMax = dfValueMax.drop(['Status'])
    return dfValueMax
##print("\nThe maximum values for Normal group")
##print(maxValues(dfNormal))
##print("\nThe maximum values for Abnormal group")
##print(maxValues(dfAbnormal))

#Function to calculate the Mean features of dfNormal or dfAbnormal groups
def meanValues(dfType):
    #drop the status column first, since it contains string values and mean inly works with numbers
    dfValueMeanTemp = dfType.drop('Status', axis=1)
    dfValueMean = dfValueMeanTemp.mean()
    return dfValueMean
##print("\nThe Mean values for Normal group")
##print(meanValues(dfNormal))
##print("\nThe Mean values for Abnormal group")
##print(meanValues(dfAbnormal))

#Function to calculate the Median features of dfNormal or dfAbnormal groups
def medianValues(dfType):
    dfValueMedianTemp = dfType.drop('Status', axis=1)
    dfValueMedian = dfValueMedianTemp.median()
    return dfValueMedian
##print("\nThe Median values for Normal group")
##print(medianValues(dfNormal))
##print("\nThe Median values for Abnormal group")
##print(medianValues(dfAbnormal))

#Mode For Normal group
modeNormal = dfNormal.drop('Status', axis=1)
#get the hightst values from each column into a py tuple
modeDfNormal = modeNormal.value_counts().idxmax(axis=0)
#Save the tuple data into a pandas dataframe
panDfNormalTemp = pd.DataFrame(modeDfNormal, index=['Power_range_sensor_1','Power_range_sensor_2','Power_range_sensor_3','Power_range_sensor_4',
                             'Pressure _sensor_1','Pressure _sensor_2','Pressure _sensor_3','Pressure _sensor_4',
                            'Vibration_sensor_1','Vibration_sensor_2','Vibration_sensor_3','Vibration_sensor_4'])
panDfNormal= panDfNormalTemp.transpose()
##print("\nMode for Normal group:\n", panDfNormal.to_string(index=False))         #.to_string(index=False) prints the df without the index value
#Mode For Abnormal group
modeAbnormal = dfAbnormal.drop('Status', axis=1)
#get the hightst values from each column into a py tuple
modeDfAbnormal = modeAbnormal.value_counts().idxmax(axis=0)
#Save the tuple data into a pandas dataframe
panDfAbnormalTemp = pd.DataFrame(modeDfAbnormal, index=['Power_range_sensor_1','Power_range_sensor_2','Power_range_sensor_3','Power_range_sensor_4',
                             'Pressure _sensor_1','Pressure _sensor_2','Pressure _sensor_3','Pressure _sensor_4',
                            'Vibration_sensor_1','Vibration_sensor_2','Vibration_sensor_3','Vibration_sensor_4'])
panDfAbnormal= panDfAbnormalTemp.transpose()
##print("\nMode for Abnormal group:\n", panDfAbnormal.to_string(index=False))     #.to_string(index=False) prints the df without the index value

#variance values
def varianceValues(dfType):
    dfVarianceTemp = dfType.drop('Status', axis=1)
    dfVariance = dfVarianceTemp.var(axis='index')
    return dfVariance
##print("\nVariance on the Normal group")
print(varianceValues(dfNormal))
##print("\nVariance on the Abnormal group")
print(varianceValues(dfAbnormal))

#Box plots
sns.set_theme(style="whitegrid")
#Plots for Normal group features
#ax= sns.boxplot(x= dfNormal["Power_range_sensor_1"], x=dfAbnormal["Power_range_sensor_1"], width=0.2, orient="v")
#ax= sns.boxplot(y= dfNormal["Power_range_sensor_1"], x=dfAbnormal["Power_range_sensor_1"])
def plotting(dfplot1, dfplot2):
    dfVal1 = dfNormal[dfplot1]
    dfVal2 = dfAbnormal[dfplot2]
    totalP= pd.concat((dfVal1, dfVal2))
    return totalP
totalP = plotting("Power_range_sensor_1","Power_range_sensor_1")
ax= sns.boxplot(y= dfNormal["Power_range_sensor_1"], x=dfNormal["Status"])

plt.show()