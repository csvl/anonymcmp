import pandas as pd
from enum import Enum

def get_spans(df, partition, categorical, scale=None):
    spans = {}
    for column in df.columns:
        if column in categorical:
            span = len(df[column][partition].unique())
        else:
            span = df[column][partition].max() - df[column][partition].min()
        if scale is not None:
            span = span / scale[column]
        spans[column] = span
    return spans


def split(df, partition, column, categorical):
    dfp = df[column][partition]
    if column in categorical:
        values = dfp.unique()
        lv = set(values[:len(values) // 2])
        rv = set(values[len(values) // 2:])
        return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
    else:
        median = dfp.median()
        dfl = dfp.index[dfp < median]
        dfr = dfp.index[dfp >= median]
        return (dfl, dfr)


def is_k_anonymous(df, partition, args):
    if len(partition) < args['k']:
        return False
    return True


def diversity(df, partition, column):
    return len(df[column][partition].unique())


def is_l_diverse(df, partition, args):
    return diversity(df, partition, args['sensitive_column']) >= args['l']


def t_closeness(df, partition, column, global_freqs):
    total_count = float(len(partition))
    d_max = None
    group_counts = df.loc[partition].groupby(column)[column].agg('count')
    for value, count in group_counts.to_dict().items():
        p = count / total_count
        d = abs(p - global_freqs[value])
        if d_max is None or d > d_max:
            d_max = d
    return d_max


def is_t_close(df, partition, args):
    if not args['sensitive_column'] in args['categorical']:
        raise ValueError("this method only works for categorical values")
    return t_closeness(df, partition, args['sensitive_column'], args['global_freqs']) <= args['p']


def partition_dataset(df, feature_columns, scale, is_valid, categorical, args):
    finished_partitions = []
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns], partition, categorical, scale)
        for column, span in sorted(spans.items(), key=lambda x: -x[1]):
            lp, rp = split(df, partition, column, categorical)
            if not is_valid(df, lp, args) or not is_valid(df, rp, args):
                continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions


def build_indexes(df, categorical):
    indexes = {}
    for column in categorical:
        values = sorted(df[column].unique())
        indexes[column] = {x: y for x, y in zip(values, range(len(values)))}
    return indexes


def agg_categorical_column(series):
    return [','.join(set(series))]


def agg_numerical_column(series):
    return series.mean()


def build_anonymized_dataset(df, partitions, feature_columns, categorical, max_partitions=None):
    aggregations = {}
    for column in feature_columns:
        if column in categorical:
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column

    df_out = df.copy(deep=True)

    for i, partition in enumerate(partitions):
        if max_partitions is not None and i > max_partitions:
            break

        if len(categorical.intersection(feature_columns)) == 0:
            grouped_columns = df.loc[partition].agg(aggregations, squeeze=False)
            values = grouped_columns.to_dict()
        else:
            values = {**{col: ','.join((str(c) for c in df.loc[partition][col].unique())) for col in
                         categorical.intersection(feature_columns)},
                      **{col: df.loc[partition][col].mean() for col in set(feature_columns) - categorical}}

        for k, v in values.items():
            df_out.loc[partition, k] = v

    return df_out

class MondrianOption(Enum):
    Non = 1
    ldiv = 2
    tclose = 3


def anonymize(data, categorical, QI, sensitive_column, k, option=MondrianOption.Non, l_or_p=1):
    assert (type(option) is MondrianOption)

    df = pd.DataFrame(data.values, columns=data.columns) if type(data) is pd.DataFrame else pd.DataFrame(data)

    full_spans = get_spans(df, df.index, categorical)

    if option == MondrianOption.Non:
        finished_partitions = partition_dataset(df, QI, full_spans,
                                                lambda *args: is_k_anonymous(*args), categorical,
                                                {'sensitive_column': sensitive_column, 'k': k, 'l': l_or_p})
    elif option == MondrianOption.ldiv:
        finished_partitions = partition_dataset(df, QI, full_spans,
                                                lambda *args: is_k_anonymous(*args) and is_l_diverse(*args), categorical,
                                                {'sensitive_column': sensitive_column, 'k': k, 'l': l_or_p})
    else: #t-closeness
        global_freqs = {}
        total_count = float(len(df))
        group_counts = df.groupby(sensitive_column)[sensitive_column].agg('count')
        for value, count in group_counts.to_dict().items():
            p = count / total_count
            global_freqs[value] = p

        finished_partitions = partition_dataset(df, QI, full_spans,
                                                lambda *args: is_k_anonymous(*args) and is_t_close(*args), categorical,
                                                {'sensitive_column': sensitive_column, 'categorical': categorical,
                                                 'k': k, 'global_freqs': global_freqs, 'p': l_or_p})

    return build_anonymized_dataset(df, finished_partitions, QI, categorical)
