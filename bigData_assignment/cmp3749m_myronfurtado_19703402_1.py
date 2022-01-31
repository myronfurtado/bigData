#import python packages that are needed
from operator import index
from turtle import color
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
##print("\nThe Mean values for Normal group: ")
##print(meanNormal)
##print("\nThe Mean values for Abnormal group")
##print(meanAbnormal)


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
print("\n")


#variance values
varianceNormal = normalDropDF.var()
varianceAbnormal = abnormalDropDF.var()
##print("\nVariance on the Normal group: ")
##print(varianceNormal)
##print("\nVariance on the Abnormal group: ")
##print(varianceAbnormal)


#(Boxplots) for the Features of Normal and Abnormal groups
#Function to get the required feature column from both groups into a single dataframe
def plotVals(col1, col2):
    #for Normal DataFrame only select the required column and rename that column to Normal for the plot
    normalColTemp = dfNormal.filter(items=[col1], axis=1)
    normalCol = normalColTemp.rename(columns={col1: 'Normal'})
    #for Abnormal DataFrame only select the required column and rename that column to Abnormal for the plot
    abnormalColTemp = dfAbnormal.filter(items=[col2], axis=1)
    abnormalCol = abnormalColTemp.rename(columns={col2: 'Abnormal'})
    
    return normalCol, abnormalCol

#loops throught each colum and generates a boxplot
## for column in pandas_df.iloc[: , 1:].columns:
##     #Calling function with required rows and combining them into a new DataFrame
##     plotDF = pd.concat(plotVals(column,column), axis=1)

##     #styling the box plots and Median line
##     plt.figure(figsize=(8,8))
##     medianprops = dict(linestyle='-', linewidth=1, color='g')
##     boxprops = dict(linestyle='-', linewidth=2, color='b')
##     #calling the boxplot
##     plotDF.boxplot(column=['Normal', 'Abnormal'], boxprops=boxprops, medianprops=medianprops)
    
##     plt.suptitle(column)
##     plt.savefig(column)
##     plt.show()

# <----------------------------------------------------------------------------------------------------------------------------------------------------->
# task 3
#correlation between Features for the full dataset
#pd.set_option('display.max_rows', None, 'display.max_columns', None) #this prints the full pandas matrix but it looks unprofessional

##plt.figure(figsize=(16,8))
##correlationMatrix = pandas_df.corr(method='pearson')
##ax = sns.heatmap(correlationMatrix, annot=True, cmap="Blues")

##ax.set_title('Correlation Heatmap')
##ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
##ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
##plt.tight_layout()
##plt.show()

#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#importing the required libraries
#stringIndexer
from pyspark.ml.feature import StringIndexer
#vectorAssember
from pyspark.sql.functions import col
#Normaliser



#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#Using StringIndexer (on "Status" column)
stringIndexer = StringIndexer(inputCol="Status", outputCol="StatusIndex", stringOrderType="alphabetAsc")
stringIndexed = stringIndexer.fit(df).transform(df)
#print
stringIndexed.select("Status","StatusIndex").sampleBy("StatusIndex", fractions={1.0: 0.02, 0.0: 0.05}).show(n=20)

#drop the Status colum now that we have index values for it
stringIndexed = stringIndexed.drop("Status")
stringIndexed.show()

#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#Vector Assembler

#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#Normalise the values

#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#task 4
#Splitting the data into train and test samples
train, test= df.randomSplit([0.7, 0.3])

##print('Total dataset records: ', df.count())
##print('Train dataset records count: ', train.count())
##print('Test dataset records count: ', test.count())
##test.show(250)
#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#task 5

#Use string indexer transformation to change Status to a number value then drop the status column (might have to do this before splitting the data to make it easier)
#Create a Normalizer transform on the adataset as it improves the ANN performance
#could try to use a vector assembler to sumush all the columns into a single 'features' column to pass to the algorithm.
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
