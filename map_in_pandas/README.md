# pagaya-map_in_pandas

Easy python wrapper for Spark mapInPandas, applyInPandas

## Goal
Easily run legacy pandas-based python functions at scale on large amounts of data.

## applications
* Running legacy transformations, feature extraction etc.
* large scale model-evaluation
* large scale experiments and parameter tuning.
* A-B testing
* Concurrent training (e.g. xgboost)

## Spark
We Chose spark for various reaseons - see alternatives below.
Spark has two APIs related to running arbitrary pandas functions over
spark-dataframe: 
* [`mapInPandas`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.mapInPandas.html)
* [`applyInPandas`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.GroupedData.applyInPandas.html)

However using these functions is buggy and not natural for the following reasons:
* Upfront result schema is required - user is required to defone the function's result in the Spark Schema format. 
Most pandas users are not used to declare the "schema" they return from function.
* debugging your code - when running in Spark your function 
is executed on remote machine - it is convenient to have a way to devug it locally* Bug when column name contains `.` 
, even if enclosed with ` - and we have such columns becasue of legacy data formats
* Bug in arrow when groupping by string fields
* Handling pandas index in the results
* Handling integer column names (which can occure in pandas)

## Scaling alternatives
You can scale your pandas/sklearn workloads in several ways
* joblib (single machine, multi-core)
* Dask - python native cluster
* Spark - JVM+Python framework - very mature with many integrations to the data processing world and tools.
* Spark-pandas (Spark 3.2 - previously named Databricks-Koalas) - they state 70% of the pandas functionality exits
However our trials at the point of writing found the framework imature and buggy.