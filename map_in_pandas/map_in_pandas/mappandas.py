__author__ = "Tal Franji, Pagaya Inc."

import traceback
from typing import Optional
import numpy as np
import pandas as pd
import pyspark
import re
from typing import cast, Any, Callable, Iterable, List, Union, Tuple

import pandas as pd
import pandas.errors
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *

# Type to allow column name, column and list of strings or columns
GROUP_BY = Union[List[Union[str, pyspark.sql.Column]], Union[str, pyspark.sql.Column]]
# Type for a function accepting pd.DataFrame and returning pd.DataFrame
PD_FUNC = Callable[[pyspark.sql.DataFrame], pyspark.sql.DataFrame]


def _wrap_field(field_name: str) -> str:
    if field_name.startswith("`"):
        return field_name  # assume already wrapped
    if re.search(r"[\W\.]", field_name):
        return f"`{field_name}`"
    return field_name


def _spark_dotbug_wrap(df: pyspark.sql.DataFrame) -> Optional[pyspark.sql.DataFrame]:
    """If column names contain '.' return a fixed dataframe with ~ instead of .
    otherwise - return None (caller should use original dataframe)"""
    if not any(["." in col for col in df.columns]):
        return None
    expressions = list()
    for col in df.columns:
        if "." in col:
            expressions.append(f"{_wrap_field(col)} as {_wrap_field(col.replace('.', '~'))}")
        else:
            expressions.append(_wrap_field(col))
    return df.selectExpr(*expressions)


PD_TYPE_TO_SPARK = {
    "object": "STRING",
    "int64": "BIGINT",
    "int32": "INT",
    "int16": "SHORTINT",
    "int8": "TINYINT",
    "float32": "FLOAT",
    "float64": "DOUBLE",
    "bool": "BOOLEAN",
    "string": "STRING",
    "datetime64[ns]": "TIMESTAMP",
}


def _pd_type_to_spark(pd_type_name: str) -> str:
    if pd_type_name in PD_TYPE_TO_SPARK:
        return PD_TYPE_TO_SPARK[pd_type_name]
    raise ValueError("unsupported Pandas type name: " + pd_type_name)


def _pd_field_name(field: Any) -> str:
    if isinstance(field, str):
        return field
    else:
        return f"_c{field}"  # for columns with int names in pandas


def _pandas_to_spark_schema(pd_df: pd.DataFrame) -> str:
    # TODO(franji): is this better than creating a Spark-DF from the panda and taking the schema?
    pd_schema = pd_df.dtypes
    fields = [_wrap_field(_pd_field_name(name)) + " " + _pd_type_to_spark(pd_type.name) for name, pd_type in pd_schema.items()]
    return ", ".join(fields)


SPARK_INT_TYPES = {
    pyspark.sql.types.IntegerType: "0",
    pyspark.sql.types.LongType: "0L",
    pyspark.sql.types.ByteType: "0Y",
    pyspark.sql.types.ShortType: "0S",
}


def fill_na_int0(spark_df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """Since pandas int cannot contain NULL - we want to prevent them from being converted to float64
    so we put zero where there are NULLs"""
    expressions = list()
    modified = False
    for field in spark_df.schema:
        name = _wrap_field(field.name)
        field_type = type(field.dataType)
        # Check for field_type is IntegralType did not work well so we use SPARK_INT_TYPES[]
        if field.nullable and field_type in SPARK_INT_TYPES:
            zero = SPARK_INT_TYPES[field_type]
            expr = f"""CASE WHEN {name} IS NULL THEN {zero} ELSE {name} END AS {name}"""
            modified = True
        else:
            expr = name
        expressions.append(expr)
    if not modified:
        return spark_df
    return spark_df.selectExpr(*expressions)


def _spark_dotbug_unwrap(pd_df: pd.DataFrame) -> pd.DataFrame:
    """for fixing the spark dot-bug - we convert here back to the original name - puting '. ' back"""
    pd_df.columns = [col.replace("~", ".") for col in pd_df.columns]
    return pd_df


def _pandas_string_to_object(pd_df: pd.DataFrame) -> pd.DataFrame:
    """when using arrow there are some issues with Spark when dtype is defined as string"""
    # we get pyarrow error similar to what is mentioned here:
    # https://stackoverflow.com/questions/65135416/pyspark-streaming-with-pandas-udf
    # so we fix strings to object types
    in_dtypes = pd_df.dtypes
    for col in in_dtypes[in_dtypes == "string"].index.to_list():
        pd_df[col] = pd_df[col].astype("object")
    return pd_df


def pre_post_processing(
    null_to_zero: bool, orig_spark_df: pyspark.sql.DataFrame, is_group_by: bool
) -> Tuple[pyspark.sql.DataFrame, PD_FUNC, PD_FUNC]:
    """given the input `orig_spark_df` return the dataframe after pre-processing.
    Also return the fixup function needed to be done before activating the user-given-function.
    The preprocessing may include 2 stages:
    1. if null_to_zero - replace integer NULL with zeros.
    2. if any column name contains "." - which triggers a bug inside mapInPandas - replace it.
    fixup is needed to "undo" the replacement of "."
    We also return a third value - the post_procesing function - this is needed
    to replace 'string' dtype to 'object' in result pandas - because it creates issue with pyarrow.
    """

    modified_spark_df = orig_spark_df
    if null_to_zero:
        modified_spark_df = fill_na_int0(orig_spark_df)
    df_dotbug = _spark_dotbug_wrap(modified_spark_df)
    if df_dotbug is None:
        pd_fixup = lambda pd_df: pd_df  # do nothing
    else:
        modified_spark_df = df_dotbug
        pd_fixup = _spark_dotbug_unwrap

    pd_post = _pandas_string_to_object

    return modified_spark_df, pd_fixup, pd_post


def pandas_is_naive_index(pd_df: pd.DataFrame) -> bool:
    """
    Check whether pd_df has a specific index or the naive one.
    """
    # Pandas doesn't have a function to check if a specific index was assigned to the dataframe.
    # The naive(basic) dataframe is the range index.
    return pd_df.index.name is None and isinstance(pd_df.index, pd.RangeIndex)


def _infer_map_in_pandas_schema(
    spark_df: pyspark.sql.DataFrame,
    func_pandas_generator: Callable[[Iterable[pd.DataFrame]], Iterable[pd.DataFrame]],
    infer_row_count: int = 100,
) -> str:
    def probe_function(iterator_pd_df: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
        for pd_df in func_pandas_generator(iterator_pd_df):
            user_generated_pd_df = pd_df
            # return a dummy dataframe with a single column `user_schema` which contains
            #  the result schema as a string.
            yield pd.DataFrame(dict(user_schema=[_pandas_to_spark_schema(user_generated_pd_df)]))
            return

    schema_df = spark_df.limit(infer_row_count).mapInPandas(probe_function, schema="user_schema string")
    schema = schema_df.take(1)[0][0]
    return cast(str, schema)


def _map_in_pandas_yield(
    spark: SparkSession,
    spark_df: pyspark.sql.DataFrame,
    func_pandas_generator: Callable[[Iterable[pd.DataFrame]], Iterable[pd.DataFrame]],
    schema: Optional[str] = None,
    debug_local_row_count: int = -1,
    cache_schema: bool = False,
) -> pyspark.sql.DataFrame:
    """_map_in_pandas_yield is a low level function - it does not handle the spark_dotbug and null to zero.
    It does handle schema inference."""
    if debug_local_row_count > 0:
        # For testing debugging withoug spark in the middle
        data = spark_df.limit(debug_local_row_count).toPandas()
        result_frames = [result_df for result_df in func_pandas_generator([data])]
        return spark.createDataFrame(pd.concat(result_frames))
    if schema is None:
        schema = _infer_map_in_pandas_schema(spark_df, func_pandas_generator)
    return spark_df.mapInPandas(func_pandas_generator, schema)


def map_in_pandas(
    spark: SparkSession,
    spark_df: pyspark.sql.DataFrame,
    func_pddf: Callable[[pd.DataFrame], pd.DataFrame],
    group_by: Optional[GROUP_BY] = None,
    schema: Optional[str] = None,
    debug_local_row_count: int = -1,
    null_to_zero: bool = True,
    cache_schema: bool = False,
) -> pyspark.sql.DataFrame:
    spark_df, pd_fixup, pd_post = pre_post_processing(null_to_zero, spark_df, group_by is not None)

    fixed_func: Callable[[pd.DataFrame], pd.DataFrame] = lambda pd_df: pd_post(func_pddf(pd_fixup(pd_df)))

    def _call_generator(iter_pd_df: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
        for pd_df in iter_pd_df:
            result = fixed_func(pd_df)
            if not pandas_is_naive_index(result):
                raise Exception(
                    f"The function {func_pddf.__name__} returns a dataframe with index."
                    f" You can use the function reset_index to transform the index into columns"
                    f" before returning the result."
                )
            yield result

    if group_by is None:
        return _map_in_pandas_yield(spark, spark_df, _call_generator, schema, debug_local_row_count, cache_schema)
    return _grouped_pandas(
        spark,
        spark_df,
        group_by,
        fixed_func,
        schema=schema,
        debug_local_row_count=debug_local_row_count,
        cache_schema=cache_schema,
    )


def _column_name(col: Union[str, pyspark.sql.Column]) -> str:
    if isinstance(col, str):
        return col
    as_str = str(col)
    # There is no reasonable way to get column name in Spark...
    match_result = re.search(r"column<'(.*)'>", as_str, re.I)
    if not match_result:
        raise RuntimeError("Internal ERROR - cannot infer column name for " + as_str)
    return match_result.group(1)


def _group_by_column_names(group_by: GROUP_BY) -> Union[str, List[str]]:
    if isinstance(group_by, list):
        return [_column_name(c) for c in group_by]
    return _column_name(group_by)


class InferUserFunctionError(Exception):
    pass


def _infer_apply_in_pandas_schema(
    spark_df: pyspark.sql.DataFrame,
    grouped_by: GROUP_BY,
    func_pddf: Callable[[pd.DataFrame], pd.DataFrame],
    infer_row_count: int = 100,
) -> str:
    def probe_function(pd_df: pd.DataFrame) -> pd.DataFrame:
        try:
            user_generated_pd_df = func_pddf(pd_df)
        except Exception as ex:
            return pd.DataFrame(dict(user_schema=["### " + traceback.format_exc()]))
        return pd.DataFrame(dict(user_schema=[_pandas_to_spark_schema(user_generated_pd_df)]))

    spark_grouped_sample = spark_df.limit(infer_row_count).groupBy(grouped_by)
    schema_df = spark_grouped_sample.applyInPandas(probe_function, schema="user_schema string")
    schema = schema_df.take(1)[0][0]
    if schema.startswith("###"):
        raise InferUserFunctionError("ERROR _infer_apply_in_pandas_schema - user function exception:\n" + schema)
    return cast(str, schema)


def _safe_func_name(f: Callable[..., Any]) -> str:
    try:
        return f.__name__
    except Exception as _ex:
        return "BAD_FUNC_NO_NAME"


def _grouped_pandas(
    spark: pyspark.sql.SparkSession,
    spark_df: pyspark.sql.DataFrame,
    group_by: GROUP_BY,
    func_pddf: Callable[[pd.DataFrame], pd.DataFrame],
    schema: Optional[str] = None,
    debug_local_row_count: int = -1,
    cache_schema: bool = False,
) -> pyspark.sql.DataFrame:
    # TODO(franji) - this fails if on of the group_by column contains '.' originally
    group_by_names = _group_by_column_names(group_by)
    assert all(["." not in name for name in group_by_names])  # for dotbug
    spark_grouped_data = spark_df.groupBy(group_by)
    if debug_local_row_count > 0:
        # For testing debugging without spark in the middle
        # since we cannot just take data from grouped-data - we use the unit function
        data_pd = (
            spark_grouped_data.applyInPandas(_pandas_string_to_object, spark_df.schema)
            .limit(debug_local_row_count)
            .toPandas()
        )
        # take the first group
        groupped_pd = data_pd.groupby(group_by_names)
        for g_name, g_data in groupped_pd:
            result_pd = func_pddf(g_data)
            break
        return spark.createDataFrame(result_pd, schema=_pandas_to_spark_schema(result_pd))
    if schema is None:
        try:
            schema = _infer_apply_in_pandas_schema(spark_df, group_by, func_pddf)
        except InferUserFunctionError as ex:
            raise RuntimeError(
                f"""ERROR user function {_safe_func_name(func_pddf)} raised error -
            please run with debug_local_row_count=100 to test it.\n Inner exception:\n"""
                + str(ex)
            )
    return spark_grouped_data.applyInPandas(func_pddf, schema)


