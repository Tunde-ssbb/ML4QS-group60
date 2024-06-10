def interpolate(data, col_name, group_by_col=None):
    print(f" NaN values in {col_name} column: {data[col_name].isna().sum()}")
    if group_by_col == None:
        data[col_name] = data[col_name].interpolate()
    else:
        data[col_name] = data.groupby(group_by_col)[col_name].apply(lambda group: group.interpolate())

    return data

