## Parameters

This document specifies a variety of parameters that control the operation of **SynDiffix**.

### Changing anonymization parameters

Anonymization parameters determine how much raw data is suppressed and distorted before synthesis.
To modify default parameters, pass a custom instance of the `AnonymizationParams` class
to the `Synthesizer` object. For example:

```py
Synthesizer(df_original, anonymization_params=AnonymizationParams(layer_noise_sd=1.5))
```

The following parameters are available:

- `salt`: secret salt for the added noise; if empty, a default value is generated automatically and
  saved into the current user's config folder for future uses.

- `low_count_params`: parameters for the low-count filter.

- `outlier_count`: outlier count interval used during flattening.

- `top_count`: top count interval used during flattening.

- `layer_noise_sd`: stddev of each noise layer added to row counts.

### Changing bucketization parameters

Bucketization parameters determine how raw data is aggregated before synthesis.
To modify default parameters, pass a custom instance of the `BucketizationParams` class
to the `Synthesizer` object. For example:

```py
Synthesizer(df_original, bucketization_params=BucketizationParams(precision_limit_depth_threshold=10))
```

The following parameters are available:

- `singularity_low_threshold`: low threshold for a singularity bucket.

- `range_low_threshold`: low threshold for a range bucket.

- `precision_limit_row_fraction`: the fraction of rows needed for splitting nodes when the tree depth goes
  beyond the depth threshold; this condition is applied in addition to the low-count filter.

- `precision_limit_depth_threshold`: tree depth threshold below which nodes are split only if they pass the
  low-count filter; when above the threshold, the row fraction condition is also applied.

### Changing clustering strategy

Clustering strategy determines how columns are grouped together in the forest of trees used for
anonymization and aggregation. Anonymized buckets are harvested from those trees, synthetic
microdata tables are generated, and are then stitched together into a single output table.
To change the clustering strategy, pass an instance of a sub-class of the `ClusteringStrategy` class
to the `Synthesizer` object. For example:

```py
Synthesizer(df_original, clustering=NoClustering())
```

The following strategies are available:

- `DefaultClustering`: - general-purpose clustering strategy; columns are grouped together in order to maximize the
  chi-square dependence measurement values of the generated clusters; a main column that gets put into every cluster
  can be specified, in order to improve output quality for that specific column.

- `MLClustering`: - strategy for ML tasks; main feature columns for a target column are automatically detected and
  grouped together with the target column in the order of their ML prediction-test scores.

- `NoClustering`: - disables clustering and instead synthesizes each column independently; the microdata of
  these columns is shuffled randomly and combined into an output table.

- `SingleClustering`: - strategy that puts all columns in a single cluster; note that this will
  result in very poor performance when the number of columns is large.
