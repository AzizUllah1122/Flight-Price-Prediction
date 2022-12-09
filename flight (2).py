!pip install pyspark py4j --quiet

import pyspark
from pyspark.sql.types import *
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName("Aziz").getOrCreate()
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import col
import pyspark.sql.functions as F

from google.colab import drive
drive.mount('/content/drive')

Df=spark.read.csv("/content/economy.csv", header=True)

Df.head(3)

x=Df.count()
print(x)

#Check null values
Df.filter(Df.to.isNull()).show()

#count number of original rows
y =Df.count()
print(y)

#count number of rows after deleting duplicate rows
z =Df.dropDuplicates().count()
print(z)
z1 = y-z
print(z1)

#Dropping all the rows having duplicate values
drop =Df.dropDuplicates().dropna(how="any").dropna(how="all")
missing = y - drop.count()
print("Rows with duplicates ",missing)
missing1=drop.count()
print("rows without duplicates ",missing1)

drop.groupBy('date').count().sort("count",ascending=False).show(7)

#check flights from each city
drop.groupBy("from").count().show()

#Checking no of most expensive flights
drop.groupBy('price').count().sort("price",ascending=False).show(3)

#Checking total airlines
drop.select('airline').distinct().count()

#separating date column
separate=pyspark.sql.functions.split(drop['date'],'-')
Df1=drop.withColumn('Day',separate.getItem(0)).withColumn('Month',separate.getItem(1)).withColumn('Year',separate.getItem(2))
Df1.show(5)

#separating dep_time column
separate1=pyspark.sql.functions.split(Df1['dep_time'],':')
Df1=Df1.withColumn('Dep(Hours)',separate1.getItem(0)).withColumn('Dep(Mins)',separate1.getItem(1))
Df1.show(5)

#separating arr_time column
separate2=pyspark.sql.functions.split(Df1['arr_time'],':')
Df1=Df1.withColumn('Arr(Hours)',separate2.getItem(0)).withColumn('Arr(Mins)',separate2.getItem(1))
Df1.show()

#splitting time_taken column
separate3=pyspark.sql.functions.split(Df1['time_taken'],' ')
Df1=Df1.withColumn('Duration(hours)',separate3.getItem(0)).withColumn('Duration(mins)',separate3.getItem(1))
separate3=pyspark.sql.functions.split(Df1['Duration(hours)'],'h')
Df1=Df1.withColumn('Duration(hours)',separate3.getItem(0))
separate3=pyspark.sql.functions.split(Df1['Duration(mins)'],'m')
Df1=Df1.withColumn('Duration(mins)',separate3.getItem(0))
Df1.show()

#separating stop and price column
Df1=Df1.withColumn('stop', regexp_replace('stop', 'non-stop', '0'))
Df1=Df1.withColumn('price', regexp_replace('price', ',', ''))
splitStop=pyspark.sql.functions.split(Df1['stop'],'-')
Df1=Df1.withColumn('stop',splitStop.getItem(0))
Df1.show(5)

cl=Df1.withColumn("Day",col("Day").cast(IntegerType())).withColumn("Month",col("Month").cast(IntegerType())).withColumn("Year",col("Year").cast(IntegerType())).withColumn("Dep(Hours)",col("Dep(Hours)").cast(IntegerType())).withColumn("Dep(Mins)",col("Dep(Mins)").cast(IntegerType())).withColumn("Arr(Hours)",col("Arr(Hours)").cast(IntegerType())).withColumn("Arr(Mins)",col("Arr(Mins)").cast(IntegerType())).withColumn("Duration(hours)",col("Duration(hours)").cast(IntegerType())).withColumn("Duration(mins)",col("Duration(mins)").cast(IntegerType())).withColumn("num_code",col("num_code").cast(IntegerType())).withColumn("stop",col("stop").cast(IntegerType())).withColumn("price",col("price").cast(IntegerType()))
cl.printSchema()
cl.show(5)

#Encoding the categorical data
from pyspark.ml.feature import StringIndexer as SI
ix=SI(inputCols=["airline","from","to"],outputCols=["Airline","From","To"])
ix = ix.fit(cl).transform(cl)
ix.show(5)

#Dropping extra columns
Data=ix.drop("date","ch_code","dep_time","time_taken","arr_time","stop","Duration(mins)")
Data.show(5)

Data=Data.select("Airline","num_code","From","To","Day","Month","Year","Dep(Hours)","Dep(Mins)","Arr(Hours)","Arr(Mins)","Duration(hours)","price")
Data.show(2)

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
Data=Data.withColumnRenamed("price","label")

fx = ['Airline','num_code','From','To','Day','Month','Year','Dep(Hours)','Dep(Mins)','Arr(Hours)','Arr(Mins)','Duration(hours)']
assembler = VectorAssembler(inputCols=fx, outputCol="features")
df = assembler.transform(Data)
df=df.select("features","label")

train, test = df.randomSplit([0.7, 0.3], seed = 3000)
rf = RandomForestRegressor(labelCol='label',featuresCol='features')
x=rf.fit(train)

pred=x.transform(test)



pred.show(4)

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.tree import RandomForest
from time import *

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
print("Root Mean Squared Error =",evaluator.evaluate(pred))

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
print("R2 =",evaluator.evaluate(pred))

#Applying linear regression 
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
y = lr.fit(train)

#Making predictions
pred2=y.transform(test)

pred2.show(4)

#Evaluating model
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(pred2)
print("Root Mean Squared Error = %g" % rmse)

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
print("R2 =",evaluator.evaluate(pred2))