from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, IntegerType, DoubleType, StructType

spark = SparkSession.builder.appName('Basics').getOrCreate()

schema = StructType([StructField("userId", IntegerType()),StructField("movieId", IntegerType()),StructField("ratings", DoubleType()),StructField("timestamp", IntegerType())])

ratings = spark.read.csv('ratings.csv', header=True, schema=schema)
ratings.show(6)

(training, test) = ratings.randomSplit([0.8, 0.2])

als = ALS(userCol="userId", itemCol="movieId", ratingCol="ratings", coldStartStrategy="drop", nonnegative = True)

param_grid = ParamGridBuilder().addGrid(als.rank, [12, 13, 14]).addGrid(als.maxIter, [18, 19, 20]).addGrid(als.regParam, [.17, .18, .19]).build()

evaluator = RegressionEvaluator(metricName="rmse", labelCol="ratings", predictionCol="prediction")

tvs = TrainValidationSplit(estimator = als, estimatorParamMaps = param_grid, evaluator = evaluator)

model = tvs.fit(training)

best_model = model.bestModel

predictions = best_model.transform(test)
rmse = evaluator.evaluator(predictions)

print(rmse)