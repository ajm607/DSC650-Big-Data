# Name: Adam McMahan
# Date: 23 Nov 2025
# Subject: Assignment 11 - DSC 650

# Libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import happybase

# Step 1: Create a Spark session

spark = SparkSession.builder.appName("HRDataset").enableHiveSupport().getOrCreate()

# Step 2: Load the data from the Hive table 'HRDataset_v13a.csv' intoa Spark DataFrame

hr_df = spark.sql(
    "SELECT EmpID, DeptID, PerfScoreID, PayRate, Position, EmpSatisfaction FROM HRDataset_v13a")

# Step 3: Handle null values by either dropping or filling them

hr_df_df = hr_df.na.drop() # Drop rows with null values

# Step 4: Prepare the data for MLlib by assembling features into a vector

assembler = VectorAssembler(
    inputCols=["PerfScoreID", "EmpSatisfaction"],
    outputCol="features",
    handleInvalid="skip" # Skip rows with null values
)
assembled_df =assembler.transform(hr_df).select("features", "PayRate")

# Step 5: Split the data into training and testing sets

train_data, test_data = assembled_df.randomSplit([0.7, 0.3])

# Step 6: Initialize and train a Linear Regression model

lr = LinearRegression(labelCol="PayRate")
lr_model = lr.fit(train_data)

# Step 7: Evaluate the model on the test data

test_results = lr_model.evaluate(test_data)

# Step 8: Print the model performance metrics

print(f"RMSE: {test_results.rootMeanSquaredError}")
print(f"R^2: {test_results.r2}")

# ---- Write metrics to HBase with happybase (using theprovided pattern) ----

# Example data (row_key, column_family:column, value) populated with the metrics
data = [
('metrics1', 'cf:rmse',
str(test_results.rootMeanSquaredError)),
('metrics1', 'cf:r2', str(test_results.r2)),
]

# Function to write data to HBase inside each partition

def write_to_hbase_partition(partition):
    connection = happybase.Connection('master')
    connection.open()
    table = connection.table('hr_data') # Update table name
    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})
    connection.close()
    
# Parallelize data and apply the function with for each Partition

rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

# Step 9: Stop the Spark session
spark.stop()