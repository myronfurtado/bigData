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

#dropping Status Column
normalDropDF = dfNormal.drop(['Status'], axis=1)
abnormalDropDF = dfAbnormal.drop(['Status'], axis=1)
#print(abnormalDropDF)

#calculate the Minimum for features of dfNormal or dfAbnormal groups using df.min() Pandas function
minNormal = normalDropDF.min()
minAbnormal= abnormalDropDF.min()
#Used to print the value of the return statement. 
##print("The minimum values for Normal group: ")
##print(minNormal)
#Print the minimum value for Abnormal group
##print("\nThe minimum values for Abnormal group: ") 
##print(minAbnormal)


#calculate the Mean features of dfNormal or dfAbnormal groups using df.max() Pandas function
maxNormal = normalDropDF.max()
maxAbnormal = abnormalDropDF.max()
##print("\nThe maximum values for Normal group: ")
##print(maxNormal)
##print("\nThe maximum values for Abnormal group: ")
##print(maxAbnormal)


#calculate the Mean features of dfNormal or dfAbnormal groups using df.mean() Pandas function
meanNormal = normalDropDF.mean()
meanAbnormal = abnormalDropDF.mean()
print("\nThe Mean values for Normal group: ")
print(meanNormal)
print("\nThe Mean values for Abnormal group")
print(meanAbnormal)


#calculate the Median features of dfNormal or dfAbnormal groups using the df.median() Pandas function
medianNormal = normalDropDF.median()
medianAbnormal = abnormalDropDF.median()
##print("\nThe Median values for Normal group: ")
##print(medianNormal)
##print("\nThe Median values for Abnormal group")
##print(medianAbnormal)


#Mode For Normal group
#create alist to append the mode values to
temp= []
#go through each colum and check its highest repeating number
for i in normalDropDF.columns:
    temp.append(normalDropDF[i].value_counts().idxmax())
##print("\nMode for Normal: ", temp)
#Mode For Abnormal group
temp=[]
#go through each colum and check its highest repeating number
for i in abnormalDropDF.columns:
    temp.append(abnormalDropDF[i].value_counts().idxmax())
##print("\nMode for Abnormal: ", temp)


#variance values
varianceNormal = normalDropDF.var()
varianceAbnormal = abnormalDropDF.var()
##print("\nVariance on the Normal group: ")
##print(varianceNormal)
##print("\nVariance on the Abnormal group: ")
##print(varianceAbnormal)


#Plotting the Features for Normal and Abnormal groups using boxplots
sns.set_theme(style="whitegrid")
#caluclate the DataFrame
#Function to get the required feature column from both groups into a single dataframe
def plotVals(col1, col2):
    #for Normal DataFrame only select the required column and rename that column to Normal for the plot
    normalColTemp = dfNormal.filter(items=[col1], axis=1)
    normalCol = normalColTemp.rename(columns={col1: 'Normal'})
    #for Abnormal DataFrame only select the required column and rename that column to Abnormal for the plot
    abnormalColTemp = dfAbnormal.filter(items=[col2], axis=1)
    abnormalCol = abnormalColTemp.rename(columns={col2: 'Abnormal'})
    
    return normalCol, abnormalCol

#create new Dataframe with the required Normal and Abnormal feature
#calling function with required rows and combining them into a new DataFrame
plotDF1 = pd.concat(plotVals('Power_range_sensor_1', 'Power_range_sensor_1'), axis=1)
#print(plotDF1)
#plot for feature 2
plotDF2 = pd.concat(plotVals('Power_range_sensor_2', 'Power_range_sensor_2'), axis=1)
#print(plotDF2)
plotDF3= pd.concat(plotVals('Power_range_sensor_3 ', 'Power_range_sensor_3 '), axis=1)
plotDF4= pd.concat(plotVals('Power_range_sensor_4', 'Power_range_sensor_4'), axis=1)
plotDF5= pd.concat(plotVals('Pressure _sensor_1', 'Pressure _sensor_1'), axis=1)
plotDF6= pd.concat(plotVals('Pressure _sensor_2', 'Pressure _sensor_2'), axis=1)
plotDF7= pd.concat(plotVals('Pressure _sensor_3', 'Pressure _sensor_3'), axis=1)
plotDF8= pd.concat(plotVals('Pressure _sensor_4', 'Pressure _sensor_4'), axis=1)
plotDF9= pd.concat(plotVals('Vibration_sensor_1', 'Vibration_sensor_1'), axis=1)
plotDF10= pd.concat(plotVals('Vibration_sensor_2', 'Vibration_sensor_2'), axis=1)
plotDF11= pd.concat(plotVals('Vibration_sensor_3', 'Vibration_sensor_3'), axis=1)
plotDF12= pd.concat(plotVals('Vibration_sensor_4', 'Vibration_sensor_4'), axis=1)

#Code for the box plots
#sns.set_theme(style="whitegrid")
#counter= np.arange(12)
#for i in counter:
#    i += 1
#    plotDF1.boxplot(column=['Normal', 'Abnormal'])
#    #plt.show()
#    plotDF2.boxplot(column=['Normal', 'Abnormal'])
#    #plt.show()
#    if i == 11:
#        break
#plt.show()

#Code for the box plots (use this unless the loop works)
#plotDF1.boxplot(column=['Normal', 'Abnormal'])
#plt.suptitle('Power_range_sensor_1')
#plt.show()
#plt.figure()
#plotDF2.boxplot(column=['Normal', 'Abnormal'])
#plt.suptitle('Power_range_sensor_2')
#plt.show()
#plotDF3.boxplot(column=['Normal', 'Abnormal'])
#plt.suptitle('Power_range_sensor_3 ')
#plt.show()
#plotDF4.boxplot(column=['Normal', 'Abnormal'])
#plt.suptitle('Power_range_sensor_4')
#plt.show()
#plotDF5.boxplot(column=['Normal', 'Abnormal'])
#plt.suptitle('Pressure_sensor_1')
#plt.show()
#plotDF6.boxplot(column=['Normal', 'Abnormal'])
#plt.suptitle('Pressure_sensor_2')
#plt.show()
#plotDF7.boxplot(column=['Normal', 'Abnormal'])
#plt.suptitle('Pressure_sensor_3')
#plt.show()
#plotDF8.boxplot(column=['Normal', 'Abnormal'])
#plt.suptitle('Pressure_sensor_4')
#plt.show()
#plotDF9.boxplot(column=['Normal', 'Abnormal'])
#plt.suptitle('Vibration_sensor_1')
#plt.show()
#plotDF10.boxplot(column=['Normal', 'Abnormal'])
#plt.suptitle('Vibration_sensor_2')
#plt.show()
#plotDF11.boxplot(column=['Normal', 'Abnormal'])
#plt.suptitle('Vibration_sensor_3')
#plt.show()
#plotDF12.boxplot(column=['Normal', 'Abnormal'])
#plt.suptitle('Vibration_sensor_4')
#plt.show()

#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#task 3
#correlation between Features for the full dataset
#pd.set_option('display.max_rows', None, 'display.max_columns', None) #this prints the full pandas matrix but it looks unprofessional

correlationMatrix = pandas_df.corr(method='pearson')
##print(correlationMatrix)

#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#task 4
#Splitting the data into train and test samples
train, test= df.randomSplit([0.7, 0.3])

##print('Total dataset records: ', df.count())
##print('Train dataset records count: ', train.count())
##print('Test dataset records count: ', test.count())
#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#task 5
#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#task 6
#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#task 7
#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#task 8
#<----------------------------------------------------------------------------------------------------------------------------------------------------->

#<----------------------------------------------------------------------------------------------------------------------------------------------------->

#<----------------------------------------------------------------------------------------------------------------------------------------------------->
# %%
