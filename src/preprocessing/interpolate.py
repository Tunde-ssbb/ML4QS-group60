# interpolate the data in columns 'col_names', group by column 'group_by_col'
def interpolate(data, col_names, group_by_col=None):
    
    for col in col_names:
        print(f" NaN values in {col} column: {data[col].isna().sum()}")
        if data[col].isna().sum() != 0:
            if group_by_col == None:
                data[col] = data[col].interpolate()
            else:
                data[col] = data.groupby(group_by_col)[col].apply(lambda group: group.interpolate())
        print(f" NaN values in {col} column: {data[col].isna().sum()}")

    return data
