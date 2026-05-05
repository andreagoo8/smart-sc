## What to use as index
The row number of the original excel: `df.index = df.index + 2`

Or use the `pseudopaz` number as index: `df = df.set_index('pseudopaz')`

## Exclude columns
Directly when reading xlsx
```
columns_to_not_use = [
    '<column_name_1>',
    '<column_name_2>',
]

usecols=lambda col: col not in columns_to_not_use
```

After reading the xlsx:
```
columns_to_delete = [
    '<column_name_1>',
    '<column_name_2>',
]
dati_escribe.drop(columns_to_delete, axis=1, inplace=True)
```

## Change dtype of a column
At import time, add `dtype={'column': 'Int64'}` for a single column and
```
int_cols = [
    <...>
]

... dtype={col: 'Int64' for col in int_cols} ...
```

After import time, run `df['col'] = df['col'].astype('Int64')`

## Exclude the time information and keep only the date one
`df['<column_name>'] = df['<column_name>'].dt.normalize()`

## Inconsistency between pseudo-patients and other columns
* `df.groupby('pseudopaz')` groups the df by the `pseudopaz` column: every subsequent operation is applied per unique pseudopaz
* `['<other>']` selects only the `<other>` column from each group
* `.nunique()` for each group, counts the number of distinct values in `<other>`. Return a Series with the `pseudopaz` as the index, and the number of distinct `<other>` values as the values
* `== 1` boolean transformation that gets `True` if the row has `1` as value, `False` otherwise
* `.all()` returns `True` if every element in the Series is `True`, `False` if even one row has a value different than one.

## Convert all sex values across sheets to consistent values
%TODO
