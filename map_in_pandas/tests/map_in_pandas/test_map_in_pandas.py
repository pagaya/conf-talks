import math
import numpy as np
import pandas as pd
import pyspark
import re
from typing import cast, Any, Callable, Iterable, List, Sequence, Union, Tuple

import map_in_pandas.mappandas as mappandas


def _spark_type_to_dtype(spark_type: str) -> str:
    """Map spark types to Pandas' """
    type_to_dtype = {
        "int": "int32",
        "byte": "int8",
        "short": "int16",
        "long": "int64",
        "tinyint": "int8",
        "shortint": "int16",
        "bigint": "int64",
        "float": "float32",
        "double": "float64",
        "string": "object",
        "boolean": 'bool',
    }
    return type_to_dtype[spark_type.lower()]


GIGA = 1024 ** 3
TEST_DATA = [
    (1, 10, 100.1234, 1000.123, 1 * GIGA, 0x7F, None, 11, 7, True),
    (None, 20, 200.1234, math.nan, 2 * GIGA, 0x41, "Key1", None, 7, False),
    (3, 30, None, None, 3 * GIGA, None, "Key1", 31, 7, True),
    (None, 40, 400.1234, 4000.123, None,       -2, "Key2",   41, 7, False),
]
TEST_SCHEMA = """
        i int, j int NOT NULL, 
        x double, y float, k bigint,
        b byte, s string,
        `m.id` short, 
        `special[0]/name.x` int,
        pred boolean""".strip()


def create_numeric_dataframe(spark: pyspark.sql.SparkSession) -> pyspark.sql.DataFrame:
    return spark.createDataFrame(TEST_DATA, schema=TEST_SCHEMA)


def create_numeric_panda() -> pd.DataFrame:
    schema_items = re.split(r"\s*,\s*", TEST_SCHEMA)
    column_names = [re.split(r"\s+", s)[0] for s in schema_items]
    spark_types = [re.split(r"\s+", s)[1] for s in schema_items]
    pandas_dtypes = [_spark_type_to_dtype(s) for s in spark_types]
    # convert None to zero in data
    data_zero = [tuple([0 if x is None else x for x in tup]) for tup in TEST_DATA]
    arr = np.array(data_zero, dtype=list(zip(column_names, pandas_dtypes)))
    return pd.DataFrame(arr, columns=column_names)


def _create_spark_session_for_test() -> pyspark.sql.SparkSession:
    builder = pyspark.sql.SparkSession.builder.appName("unittest_map_in_panads")
    builder = builder.master("local[*]")
    return builder.getOrCreate()


spark = _create_spark_session_for_test()


def test_map_in_pandas_spark_bug() -> None:
    """reproduce spark's bug with a column that has . inside ``
    if/when the bug in Spark is fixed - this test will fail and we can remove the workaround."""
    df = create_numeric_dataframe(spark)

    def debug_pd_df_iter(pd_df_iterator: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
        for i, pd_df in enumerate(pd_df_iterator):
            yield pd.DataFrame(dict(status=["OK"]))

    try:
        df.mapInPandas(debug_pd_df_iter, schema="status string")
        ok = False  # we are expecting an exception
    except pyspark.sql.utils.AnalysisException as _ex:
        # df.mapInPandas fails on:
        # pyspark.sql.utils.AnalysisException: Cannot resolve column name "special[0]/name.x" among
        # (i, j, x, y, k, b, s, m, special[0]/name.x, pred); did you mean to quote the `special[0]/name.x` column?
        # The bug is in the column conversion between spark dataframe to pandas dataframe.
        ok = True  # ok - we're expecting this 0 this is the bug
    assert ok
    # The following lines are workaround for treating the bug(They are part of map_in_pandas in yoshi_map_pandas.py),
    # they alter . to ~ in the column names.
    # The wrapped columns dataframe becomes: ['i', 'j', 'x', 'y', 'k', 'b', 's', 'm', 'special[0]/name~x', 'pred']
    column_wrapped_spark_df = mappandas._spark_dotbug_wrap(df)
    column_wrapped_spark_df.mapInPandas(debug_pd_df_iter, schema="status string")


def test_arrow_issue() -> None:
    # we get pyarrow error similar to what is mentioned here:
    # https://stackoverflow.com/questions/65135416/pyspark-streaming-with-pandas-udf
    # We handle this bug in  the pd_post function returned by pre_post_processing.
    df = create_numeric_dataframe(spark)
    df = df.drop('special[0]/name.x')  # avoid another bug we are handling in test_map_in_pandas_spark_bug
    df = df.drop('m.id')
    # if dtype= is given as 'object' the pyarrow bug does not happen.
    func = lambda pd_df: pd.DataFrame([pd_df.s.astype(str).max()], columns=["max_s"], dtype="object")
    max_str_df = mappandas._grouped_pandas(spark, df, ["s"], func)
    assert max_str_df.toPandas().max_s.to_list() == ["None", "Key1", "Key2"]
    # if function returns a string - there is a pyarrow exception - this is what we handle in the code
    func_bad = lambda pd_df: pd.DataFrame([pd_df.s.astype(str).max()], columns=["max_s"], dtype="string")
    max_str_df = mappandas._grouped_pandas(spark, df, ["s"], func_bad)
    try:
        assert max_str_df.toPandas().max_s.to_list() == ["None", "Key1", "Key2"]
        ok = False  # we expect an exception because of pyarrow bug
    except:
        ok = True
    assert ok


def test_pandas_to_spark_schema() -> None:
    df = create_numeric_panda()
    schema = mappandas._pandas_to_spark_schema(df)
    assert (
        schema.lower() == "i INT, j INT, "
        "x DOUBLE, y FLOAT, k BIGINT, b TINYINT, s STRING, "
        "`m.id` SHORTINT, `special[0]/name.x` int, pred boolean".lower()
    )


def test_map_in_pandas() -> None:
    df = create_numeric_dataframe(spark)
    sum_df = mappandas.map_in_pandas(spark,
                                     df, lambda pd_df: pd.DataFrame([pd_df.j.sum()], columns=["sum_j"]))
    assert sum_df.toPandas().sum_j.sum() == 10 + 20 + 30 + 40
    # try again but with run local
    sum_df2 = mappandas.map_in_pandas(spark,
        df, lambda pd_df: pd.DataFrame([pd_df.j.sum()], columns=["sum_j2"]), debug_local_row_count=10
    )
    assert sum_df2.toPandas().sum_j2.sum() == 10 + 20 + 30 + 40
    sum_df3 = mappandas.map_in_pandas(spark,
        df, lambda pd_df: pd.DataFrame([pd_df["special[0]/name.x"].sum()], columns=["sum_dot"])
    )
    assert sum_df3.toPandas().sum_dot.sum() == 7 * 4
    sum_df4 = mappandas.map_in_pandas(spark,
        df, lambda pd_df: pd.DataFrame([pd_df["m.id"].sum()], columns=["m.id"])
    )
    assert sum_df4.toPandas()["m.id"].sum() == 11 + 31 + 41
    max_str_df = mappandas.map_in_pandas(spark,
                                         df, lambda pd_df: pd.DataFrame([pd_df.s.astype(str).max()], columns=["max_s"]))
    assert max_str_df.toPandas().max_s.min() == "Key1"


def test_map_in_pandas_grouped() -> None:
    df = create_numeric_dataframe(spark)

    def _sum_by_key(pd_df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame([(pd_df.s.iloc[0], pd_df.j.sum())], columns=["s", "sum_j"])
        return result

    # test in local mode
    group_df_local = mappandas.map_in_pandas(spark, df, _sum_by_key, group_by=df.s, debug_local_row_count=100)
    group_pd_df_local = group_df_local.toPandas()
    assert group_pd_df_local[group_pd_df_local.s == "Key1"].iloc[0].sum_j == 50
    # now test in Spark mode
    group_df = mappandas.map_in_pandas(spark, df, _sum_by_key, group_by=df.s)
    group_pd_df = group_df.toPandas()
    assert group_pd_df[group_pd_df.s == "Key1"].iloc[0].sum_j == 50
    # test in Spark mode - group by column as a list of strings
    group_df = mappandas.map_in_pandas(spark, df, _sum_by_key, group_by=["s"])
    group_pd_df = group_df.toPandas()
    assert group_pd_df[group_pd_df.s == "Key1"].iloc[0].sum_j == 50
    # test special names with dot etc.

    def _sum_dot(pd_df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame([(pd_df.s.iloc[0], pd_df["special[0]/name.x"].sum())], columns=["s", "sum_dot"])
        return result

    group_df = mappandas.map_in_pandas(spark, df, _sum_dot, group_by=["s"])
    group_pd_df = group_df.toPandas()
    assert group_pd_df[group_pd_df.s == "Key1"].iloc[0].sum_dot == 14
