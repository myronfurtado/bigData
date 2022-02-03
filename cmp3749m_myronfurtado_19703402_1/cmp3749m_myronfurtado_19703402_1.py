#import python packages that are needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql.functions import col
from pyspark.sql.types import StructType,StructField,StringType,IntegerType
from pyspark.sql.types import IntegerType
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
import findspark
findspark.init()

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
##df.show(n=20, truncate=50)
##df.printSchema()

rows= str(df.count())
print("The datset is made of: "+ rows + " rows \n")

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
print("Missing/Invalid values in the dataset: ")
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


#calculate the Minimum for features of dfNormal or dfAbnormal groups using df.min() Pandas function
minNormal = normalDropDF.min()
minAbnormal= abnormalDropDF.min()
#Used to print the value of the return statement. 
print("The minimum values for Normal group: ")
print(minNormal)
#Print the minimum value for Abnormal group
print("\nThe minimum values for Abnormal group: ") 
print(minAbnormal)


#calculate the Mean features of dfNormal or dfAbnormal groups using df.max() Pandas function
maxNormal = normalDropDF.max()
maxAbnormal = abnormalDropDF.max()
print("\nThe maximum values for Normal group: ")
print(maxNormal)
print("\nThe maximum values for Abnormal group: ")
print(maxAbnormal)


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
print("\nThe Median values for Normal group: ")
print(medianNormal)
print("\nThe Median values for Abnormal group")
print(medianAbnormal)


#Mode For Normal group
#create alist to append the mode values to
temp= []
#go through each colum and check its highest repeating number
for i in normalDropDF.columns:
    temp.append(normalDropDF[i].value_counts().idxmax())
print("\nMode for Normal: ", temp)
#Mode For Abnormal group
temp=[]
#go through each colum and check its highest repeating number
for i in abnormalDropDF.columns:
    temp.append(abnormalDropDF[i].value_counts().idxmax())
print("\nMode for Abnormal: ", temp)


#variance values
varianceNormal = normalDropDF.var()
varianceAbnormal = abnormalDropDF.var()
print("\nVariance on the Normal group: ")
print(varianceNormal)
print("\nVariance on the Abnormal group: ")
print(varianceAbnormal)


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
for column in pandas_df.iloc[: , 1:].columns:
    #Calling function with required rows and combining them into a new DataFrame
    plotDF = pd.concat(plotVals(column,column), axis=1)

    #styling the box plots and Median line
    plt.figure(figsize=(8,8))
    medianprops = dict(linestyle='-', linewidth=1, color='g')
    boxprops = dict(linestyle='-', linewidth=2, color='b')
    #calling the boxplot
    plotDF.boxplot(column=['Normal', 'Abnormal'], boxprops=boxprops, medianprops=medianprops)
    
    plt.suptitle(column)
    #plt.savefig(column)
    plt.show()

# <----------------------------------------------------------------------------------------------------------------------------------------------------->
# task 3
#correlation between Features for the full dataset
#pd.set_option('display.max_rows', None, 'display.max_columns', None) #this prints the full pandas matrix but it looks unprofessional

plt.figure(figsize=(16,8))
correlationMatrix = pandas_df.corr(method='pearson')
ax = sns.heatmap(correlationMatrix, annot=True, cmap="Blues")

ax.set_title('Correlation Heatmap')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
plt.tight_layout()
plt.savefig("Correlation Heatmap")
plt.show()

#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#task 4

#Splitting the data into train and test samples
train, test= df.randomSplit([0.7, 0.3])

trainN =train.filter(col("Status")=="Normal").count()
trainAn=train.filter(col("Status")=="Abnormal").count()
trainTotal=trainN+trainAn
print("\nTrain (Normal group) examples count: ",trainN)
print("Train (Abormal group) examples count: ",trainAn)
print('Train dataset total records count: ', trainTotal)

testN =test.filter(col("Status")=="Normal").count()
testAn=test.filter(col("Status")=="Abnormal").count()
testTotal=testN+testAn
print("\nTest (Normal group) examples count: ",testN)
print("Test (Abormal group) examples count: ",testAn)
print('Test dataset total records count: ', testTotal)
print("\n")

#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#task 5

#importing the required libraries
#stringIndexer
from pyspark.ml.feature import StringIndexer
#vectorAssember
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
#Normaliser
from pyspark.ml.feature import Normalizer
#Pipline and classifiers
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import MultilayerPerceptronClassifier

#Perform transforms on the train dataset
#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#Using StringIndexer (on "Status" column)
stringIndexer = StringIndexer(inputCol="Status", outputCol="statusLabel", stringOrderType="alphabetAsc")
stringIndexed = stringIndexer.fit(train).transform(train)
print("String Indexer-(statusLabel) output: ")
stringIndexed.select("Status","statusLabel").sampleBy("statusLabel", fractions={0.0: 0.03, 1.0: 0.03}).show(n=25)

#drop the Status colum now that we have index values for it
stringIndexed = stringIndexed.drop("Status")

#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#Vector Assembler
vAssembler = VectorAssembler(inputCols=["Power_range_sensor_1","Power_range_sensor_2","Power_range_sensor_3 ","Power_range_sensor_4",
                                        "Pressure _sensor_1","Pressure _sensor_2","Pressure _sensor_3","Pressure _sensor_4",
                                        "Vibration_sensor_1","Vibration_sensor_2","Vibration_sensor_3","Vibration_sensor_4"],
                                        outputCol="features")
assembledVector = vAssembler.transform(stringIndexed)
#assembledVector.printSchema()
#Drop all columns other than "features" and "Status Index"
assembledVector = assembledVector.drop("Power_range_sensor_1","Power_range_sensor_2","Power_range_sensor_3 ","Power_range_sensor_4",
                                        "Pressure _sensor_1","Pressure _sensor_2","Pressure _sensor_3","Pressure _sensor_4",
                                        "Vibration_sensor_1","Vibration_sensor_2","Vibration_sensor_3","Vibration_sensor_4")
print("assembled vector: ")
assembledVector.select("statusLabel", "features").show(n=20, truncate=False)
print("assembled vector Schema: ")
assembledVector.printSchema()

#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#Normalise the values
normalizer = Normalizer(inputCol=vAssembler.getOutputCol(), outputCol="normFeatures", p=1.0)
l1NormData = normalizer.transform(assembledVector)#print("Normalized using L^1 norm: ")
print("Normalized data: ")
l1NormData.show(truncate=False)
print("Normalizer output Schema: ")
l1NormData.printSchema()

#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#Pipeline for DecisionTree model
print("------------------------------------------------------------------------------Pipeline for Decision Tree-----------------------------------------------------------------------------")
#use output of vectorAssembler as input indtead of Normaliser as it improves accuracy by 10%
dt= DecisionTreeClassifier(featuresCol="normFeatures", labelCol="statusLabel")
#declare stages variable
stagesDecisionTree = [stringIndexer, vAssembler, normalizer, dt]
#instantiate the pipeline by passing stages variable to it
pipeline1 = Pipeline(stages=stagesDecisionTree)
#Fit the model on the training set by calline the train dataframe
pipemodelDT = pipeline1.fit(train)

#run the trained PipelineModel on the test set
predictedDecsionTree = pipemodelDT.transform(test)
##predictedDecsionTree.printSchema()

#drop non-essential columns
predictedDecsionTree = predictedDecsionTree.drop("Power_range_sensor_1","Power_range_sensor_2","Power_range_sensor_3 ","Power_range_sensor_4",
                                                 "Pressure _sensor_1","Pressure _sensor_2","Pressure _sensor_3","Pressure _sensor_4",
                                                 "Vibration_sensor_1","Vibration_sensor_2","Vibration_sensor_3","Vibration_sensor_4",
                                                "features","rawprediction","probability")

#converting 'statusLabel' and 'prediction' columns, from double to interger type
predictedDescisionTreefn= predictedDecsionTree.withColumn("statusLabel",predictedDecsionTree["statusLabel"].cast(IntegerType())).withColumn("prediction",predictedDecsionTree["prediction"].cast(IntegerType()))
print("Schema of the predicted table: ")
predictedDescisionTreefn.printSchema()
##print("Descision Tree Prediction table: ")
predictedDescisionTreefn.show()

#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#Pipeline for Support Vector Machine    
print("------------------------------------------------------------------------------Pipeline for Support Vector Machine-----------------------------------------------------------------------")
#Trainng SVC model
lsvc= LinearSVC(featuresCol='normFeatures', labelCol='statusLabel', maxIter=100, regParam=0.2) 
#Stages variable
stagesSVC = [stringIndexer, vAssembler, normalizer, lsvc]
#instantiate pipeline
pipeline2 = Pipeline(stages=stagesSVC)
#fit the pipeline model to train set
pipemodelSVC = pipeline2.fit(train)

#Run the trained PipelineModel on the test set
predictedSVC = pipemodelSVC.transform(test)

#drop non-essential columns
predictedSVC = predictedSVC.drop("Power_range_sensor_1","Power_range_sensor_2","Power_range_sensor_3 ","Power_range_sensor_4",
                                                 "Pressure _sensor_1","Pressure _sensor_2","Pressure _sensor_3","Pressure _sensor_4",
                                                 "Vibration_sensor_1","Vibration_sensor_2","Vibration_sensor_3","Vibration_sensor_4",
                                                "features","rawprediction","probability")
#predictedSVC.printSchema()

#converting 'statusLabel' and 'prediction' columns, from double to interger type
predictedSVCfn= predictedSVC.withColumn("statusLabel",predictedSVC["statusLabel"].cast(IntegerType())).withColumn("prediction",predictedSVC["prediction"].cast(IntegerType()))
print("Schema of the predicted table: ")
predictedSVCfn.printSchema()
print("Support vector machine(SVC) Prediction table: ")
predictedSVCfn.show()

#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#Pipeline for Artificial Neural Network(MLP or ANN)
print("------------------------------------------------------------------------------Pipeline for Artificial Neural Network---------------------------------------------------------------------")
#Trainng ANN model
layers = [12, 15, 2] #here first value(12) is the number of values in the featuresCol-normFeatures and last value(2) is the number of values in teh outpot col
ann = MultilayerPerceptronClassifier(featuresCol='normFeatures', labelCol='statusLabel', maxIter=500, layers=layers)
#declare stages variable
stagesANN = [stringIndexer, vAssembler, normalizer, ann]
#instantiate the pipeline by passing stages variable to it
pipeline3 = Pipeline(stages=stagesANN)
#Fit the model on the training set by calline the train dataframe
pipemodelANN = pipeline3.fit(train)

#run the trained PipelineModel on the test set
predictedANN = pipemodelANN.transform(test)
#drop non-essential columns
predictedANN = predictedANN.drop("Power_range_sensor_1","Power_range_sensor_2","Power_range_sensor_3 ","Power_range_sensor_4",
                                                 "Pressure _sensor_1","Pressure _sensor_2","Pressure _sensor_3","Pressure _sensor_4",
                                                 "Vibration_sensor_1","Vibration_sensor_2","Vibration_sensor_3","Vibration_sensor_4",
                                                "features","normFeatures","rawprediction","probability")

#converting 'statusLabel' and 'prediction' columns, from double to interger type
predictedANNfn= predictedANN.withColumn("statusLabel",predictedANN["statusLabel"].cast(IntegerType())).withColumn("prediction",predictedANN["prediction"].cast(IntegerType()))
print("Schema of the predicted table: ")
predictedANNfn.printSchema()
print("Artificial Neural Network(Multilayer perceptron classifier) Prediction table: ")
predictedANNfn.show()

#<----------------------------------------------------------------------------------------------------------------------------------------------------->
#error rate
def predictions(predicted, toSay):
    
    #calculating accuracy for Training Tree
    accuracy = predicted.filter(predicted.statusLabel == predicted.prediction).count() / float(test.count())
    print("\nThe accuracy of the", toSay, accuracy)
    
    # count how many rows have correct positive predictions
    positivecorrect = predicted.where((col("statusLabel")=="1") & (col("prediction")==1)).count()
    #print("positivecorrect: " + str(positivecorrect))
    # count how many rows have correct negative predictions
    negativecorrect = predicted.where((col("statusLabel")=="0") & (col("prediction")==0)).count()
    #print("negative correct: " + str(negativecorrect))
    # count how many rows have incorrect positive predictions
    positiveincorrect = predicted.where((col("statusLabel")=="1") & (col("prediction")==0)).count()
    #print("positive incorrect: " + str(positiveincorrect))
    # count how many rows have incorrect negative predictions
    negativeincorrect = predicted.where((col("statusLabel")=="0") & (col("prediction")==1)).count()
    #print("negative incorrect: " + str(negativeincorrect))

    #Error rate
    num1 = positiveincorrect+negativeincorrect
    print("Total Incorrectly calssified: ",num1)
    num2 = positivecorrect+negativecorrect+positiveincorrect+negativeincorrect
    print("Total Correctly calssified: ",num2)
    errorRate = num1/num2
    print("Error rate: ", errorRate)
    
    #Sensitivity(True positive Rate)
    num1 = positivecorrect+positiveincorrect
    sensitivity = positivecorrect/num1
    print("Sensitivity: ", sensitivity)
    
    #Specificity(True negative rate)
    num1 = negativecorrect+negativeincorrect
    specificity = negativecorrect/num1
    print("Specificity: ",specificity)
    

#Calling the predicted df of the given algorithm
print("------------------------------------------------------------------------------Pipeline for Decision Tree--------------------------------------------------------------------------------")
predictions(predictedDescisionTreefn, "Decision Tree classifier: ")
print("------------------------------------------------------------------------------Pipeline for Support Vector Machine-----------------------------------------------------------------------")
predictions(predictedSVCfn, "SVC classifier: ")
print("------------------------------------------------------------------------------Pipeline for Artificial Neural Network--------------------------------------------------------------------")
predictions(predictedANNfn, "Artificial Neural Network(MPC) classifier: ")
print("\n")

#<----------------------------------------------------------------------------------------------------------------------------------------------------->

#task 8
#Mapreduce
df2 = spark.read.csv("nuclear_plants_big_dataset.csv", inferSchema=True, header=True)


df3 = df2.select([count(when(col(c).contains('None') | \
                            col(c).contains('NONE') | \
                            col(c).contains('NULL') | \
                            col(c).contains('null') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in df.columns])
print("Missing/Invalid values in the dataset: ")
df3.show()

#convert spark df to pandas df
pandas_df2 = df2.toPandas()
#print(pandas_df)

#Selects the only the rows where the status is "Normal" and saves them to a variable dfNormal
dfNormalB = pandas_df2.loc[pandas_df2['Status'] == "Normal"]

#dropping Status Column
normalDropDFBig = dfNormalB.drop(['Status'], axis=1)

#calculate the Minimum for features of dfNormal or dfAbnormal groups using df.min() Pandas function
minNormalB = normalDropDFBig.min()
print("The minimum values for Normal group: ")
print(minNormalB)

#calculate the Mean features of dfNormal or dfAbnormal groups using df.max() Pandas function
maxNormalB = normalDropDFBig.max()

print("\nThe maximum values for Normal group: ")
print(maxNormalB)

#calculate the Mean features of dfNormal or dfAbnormal groups using df.mean() Pandas function
meanNormalB = normalDropDFBig.mean()
print("\nThe Mean values for Normal group: ")
print(meanNormalB)



spark.stop()
#<----------------------------------------------------------------------------------------------------------------------------------------------------->
# %%
#to stop the Sparksession to save memory
