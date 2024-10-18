## Stitching API

When building tables that have too many columns to scale to a single cluster (tree), SynDiffix builds multiple clusters and stitches them together on one or more common columns.

The API for stitching is now exposed. This is primarily for testing and development purposes: the clustering decisions made by SynDiffix are generally good and do not need to be over-ridden.

The interface for the stitching API is:

```python
from syndiffix.stitcher import stitch

df_stitched = stitch(df_left=df_left, df_right=df_right, shared=False)
```

`df_left` and `df_right` are dataframes. They must have at least one column in common. Stitching will take place on the common columns. `df_stitched` will contain the common columns as well as the non-common columns from both `df_left` and `df_right`. `df_left` and `df_right` do not need to have the same number of rows, but in practice they should not differ by more than a few rows. Otherwise, the quality of `df_stitched` will be poor (many dropped or replicated rows from `df_left` and `df_right`).

`shared` is `True` by default. If `shared==False`, then the common columns in `df_left` will be preserved in `df_stitched`: they will not be modified by the stitching procedure. If `shared==True`, then the common columns in both `df_left` and `df_right` will be modified.

Examples of stitching can be found at `tests/test_stitcher.py`.