# Step-by-step description of the algorithm

The following steps refer to the `anonymized (noisy) count` whenever a `count` is mentioned
(the `true count` is never used directly).

## Casting to real

Before forest building is conducted, preliminary casting of all values to reals has to be done:

- **null** - see [Forest building](#forest-building)
- **boolean** - false becomes 0.0, true becomes 1.0
- **integer** - natural cast to real
- **real** - kept intact (non-finite reals aren't currently handled)
- **string** - a sorted array with all of the distinct values is first created, then each value gets assigned
  the index of its position in the array
- **timestamp** - becomes seconds since 1800-01-01 00:00:00 UTC

## Forest building

There are two types of nodes: branches and leaves. Both kinds store an array of ranges, an array of subnodes, and a stub flag.
A branch has `2^tree_dimensions` children, some of which can be missing. A leaf stores the list of rows matching its ranges.

- For each row:
  - For each column:
    - Compute the range of the values space.
- For each column:
  - Map nulls to a value outside twice the actual min or max boundary,
    to prevent leaking low-count nulls into the bottom or top values range.
  - Snap the range size to the next power-of-two.
  - Align the range boundaries to the snapped size.
- For k in 1..length(columns):
  - For each combination of k columns:
    - Build the tree root for the column combination.
    - For each row:
      - Find target leaf node. If one doesn't exist:
        - Create a new leaf node by halfing the ranges of the parent.
        - Fill the subnodes list from the lower dimensional trees.
          - The subnodes are the children of parent's subnodes with the same corresponding ranges as the current node.
        - Determine if the node is a stub.
          - A node is a stub if all of its subnodes either:
            - are missing;
            - fail the range / singularity threshold;
            - are also stubs themselves.
      - Add the row to its matching leaf node.
      - If target node is not a singularity, passes LCF, is not a stub, and (below depth threshold or over row fraction):
        - Convert the target leaf node to a branch node.
        - Add the previously stored rows to the newly created node.
    - If k = 1:
      - Push the root down as long as one of its children fails LCF.
      - Add the rows from the previously dropped low-count nodes to the remaining tree.
      - Update the snapped range for the corresponding dimension.

### Bucket harvesting

A bucket has an array of ranges (one for each dimension), and a noisy row count.
When creating a bucket from a node, use the node's singularity ranges at its singularity dimensions
and its snapped ranges elsewhere.

- Do a bottom-up traversal of the forest's root. At each node:
  - If the node is a leaf:
    - If the node fails LCF:
      - Return nothing.
    - If the node is a singularity:
      - Return a bucket from node's single value.
    - If in 1-dimensional node:
      - Return a bucket from node's ranges.
    - Else:
      - Return the refined buckets from node's ranges and subnodes.
  - Else:
    - Gather all buckets from child nodes.
    - If there are fewer rows in the children's buckets than half of the current node's rows:
      - If in 1-dimensional node:
        - Return a bucket from node's ranges.
      - Else:
        - Generate `current.count - children.count` refined buckets from node's ranges and subnodes.
        - Append the refined buckets to the previously harvested buckets.
        - Return the resulting buckets.
    - Else:
      - Adjust the counts of the buckets so that their sum matches the count of the current node.
      - Return the resulting buckets.

#### Bucket refinement

- Gather the sub-dimensional buckets. For each possible subnode:
  - If the subnode's combination corresponds to a singularity:
    - Return a bucket with singularity ranges for it.
  - If the subnode exists:
    - Harvest and return the buckets for it.
  - Else:
    - Return an empty list.
- If insufficient buckets for a subnode:
  - Return the input bucket.
- Compute the possible cumulative ranges for each subnode and dimension.
- Compute the smallest ranges for all dimensions.
- Compact smallest ranges by excluding any low-count half-ranges.
- Gather the possible per-dimension ranges which are inside the smallest ranges.
- Drop the sub-dimensional buckets which are outside the smallest ranges.
- If insufficient ranges:
  - Return the input bucket.
- Else:
  - Generate new buckets by matching the remaining possible ranges of a column with
    their corresponding sub-dimensional buckets using a deterministic random shuffle.

### Microdata generation

- For each harvested bucket:
  - For 1..bucket.count:
    - For each range in the bucket:
      - Generate a random float value inside the range. In the case of a **string** column, the upper edge of the range must not exceed the distinct values sorted array.
      - Cast the generated value back to the corresponding column type.
        - In the case of **string** columns:
          - If the range is a singularity:
            - Return the exact value.
          - Else:
            - If the random value is a singularity in the corresponding 1-dimensional tree and passes LCF, return the exact value.
            - Else:
              - Return the common prefix of the range, plus `*`, plus a random integer from inside the range.
    - Return a synthetic row from the generated values.

### Clustering

When input data has a large number of dimensions, columns are divided into clusters.
The solver searches for permutations that build clusters with best overall dependence between columns.

#### Solver

- Current permutation <- {1..dimensions}.
- Best solution <- quality of initial permutation.
- Temperature <- initial temperature.
- While temperature > minimum temperature:
  - New permutation <- swap a random pair in current permutation.
  - If quality of new permutation is better than quality of current permutation,
  - Or accept with some probability depending on temperature:
    - Current permutation <- new permutation.
  - If quality of new permutation better than quality of best permutation:
    - Best solution <- current permutation.
  - Lower temperature based on a curve function.
- Return clusters built from best solution.

#### Building clusters from a permutation

Each cluster has a maximum (configurable) weight.
A column's weight is computed as `1 + sqrt(max(1, entropy))`.

- Read columns in the permutation from left to right.
- Keep adding columns to the initial cluster until:
  - The initial cluster is full,
  - Or the next column has average dependence score (to the cluster's columns) below merge threshold.
- Rest of columns are added to derived clusters. Initially derived clusters are an empty list.
- For each remaining column (left to right):
  - Find all derived clusters where:
    - Average dependence score is above merge threshold.
    - Cluster's current weight + this column's weight do not exceed 70% of max cluster weight.
  - Pick cluster with best average score.
  - If such cluster exists:
    - Add column to cluster.
  - Else:
    - Start a new cluster with this column.
- For all derived clusters (earliest to latest):
  - Look back at the columns that are available so far (including initial cluster).
  - Sort available columns by average dependence score to the derived columns (best to worst).
  - Always use first column from this list as a stitch column.
  - For each remaining column:
    - If cluster has available space and column's average dependence score is above threshold:
      - Add stitch column to cluster.
- Return initial cluster and derived clusters.

If we have a main column, it will always appear in the initial cluster, and always as the first stitch column in derived clusters.

#### Measuring clustering quality

Clustering quality is the average unsatisfied dependence between pairs of columns.
A lower value is better because more dependencies are preserved.

- Create a copy of the dependence matrix.
- For each cluster in clusters:
  - For each possible pair col1, col2 in cluster:
    - Set matrix's cells (col1,col2) to zero and (col2,col1) to zero.
- Take the sum of the matrix, excluding values in the diagonal.
- Return `sum / (2 * num_columns)`.

#### Measuring dependence between columns

Dependence between columns is a univariate measure that returns a score from 0.0 to 1.0.
SynDiffix uses a modified version of chi2 dependence measure.
The dependence matrix is the num_columns x num_columns matrix with all pairs of scores.

### Stitching

Stitching is the process of combining microdata from two or more clusters into a single table.

- Initial microtable <- synthesize initial cluster.
- For each cluster in derived clusters:
  - Produce synthetic data for cluster.
  - microtable <- stitch current microtable with cluster's microdata.
- Return the complete microtable.

By "left rows" we refer to the current microtable, and "right rows" refers to the upcoming cluster's microtable.

#### Patching

If the cluster has no stitch columns, we perform a _patch_:

- Randomly shuffle right rows.
- Align the number of right rows to be that of left rows.
  - If both sides have equal count of rows, do nothing.
  - If right has fewer rows, repeat `left.count - right.count` random rows from the right side.
  - If right has more, drop `right.count - left.count` last rows from the right side.
- Zip left and right rows:
  - For each pair, produce a microdata row that combines columns of both sides.
- Return list of combined rows.

For clusters with stitch columns, we perform a recursive _stitch_.

#### Recursive stitching

- Preparation step
  - Sort stitch (common) rows from lowest entropy to highest.
  - For each stitch column, start from the complete 1-dim range of that column.
  - Start the sorting process from the first stitch column in the sorted list.
- Sort both left and right sides by the current sort column.
- Find the midpoint of sort column's 1-dim range.
- Split both left and right rows based on the midpoint.
  - We call rows that have a value of greater than or equal to the midpoint the "upper half".
  - Rows that have value less than the midpoint are called the "lower half"
- If the lower halves of opposing microtables have a roughly equal number of rows, and same for upper halves:
  - Accept the split.
  - Move to the next sort column in the list. Wrap around if necessary.
  - Recursively stitch lower left with lower right. The nested stitch sees only the lower half of the 1-dim range.
  - Recursively stitch upper left with upper right. The nested stitch sees only the upper half of the 1-dim range.
- Else:
  - Reject the split.
  - Move to the next sort column in the list. Wrap around if necessary.
  - If all sort columns were rejected consecutively:
    - Merge left and right.
    - Emit merged rows.
  - Else:
    - Recursively stitch left and right rows.

#### Microtable merging

- Randomly shuffle left and right rows.
- Compute the average length of rows between left and right side.
- Align left and right rows to that length.
  - If a side has fewer rows, extend by randomly repeating `avg_rows - current` rows.
  - If a side has more rows, drop the last `current - avg_rows` rows.
- Perform a multi level sort on left and right rows. Use the same sort hierarchy as in the stitching phase.
- For `i` <- `1` to `avg_rows`:
  - Merge left and right rows by taking distinct columns from both sides.
  - For the shared stitch columns:
    - If either side is a "stitch owner", always use that side's stitch columns.
    - Else:
      - if `i` is odd, use the stitch columns from the left side.
      - If `i` is even, use the stitch columns from the right side.
  - Add the merged row to the resulting microtable.
- Return the microtable.

#### Clustering for ML targets

When building microdata for an ML target column, we run a simpler version of the cluster builder.

- Use sci-kit learn to find the best `k` ML features (columns with high ML relevance) for the target column.
- Create initial cluster which at first contains only the target column.
- Set initial cluster as the current cluster.
- For each feature in k-features (ordered by importance, highest to lowest):
  - If current cluster has available space:
    - Add feature to current cluster.
  - Else:
    - Start a new cluster and set it as the current cluster.
    - Add the target column as a stitch column to this cluster.
    - Add feature to current cluster.
- The first cluster is the initial cluster, the rest are derived clusters which are stitched by the target column.
- If patching is enabled:
  - For each non-feature column:
    - Add derived cluster with no stitch column and the single derived column `[column]`.
- Return clusters.

In other words, we chunk the k-features by a max size. The target column is present in each one.
We optionally patch the non-feature columns to get a complete table.

### Noise

Noise is added to entity and row counts using an RNG seeded deterministically from the bucket's contents.
This ensures identical output for the same data in different runs.

Entity threshold checks use a single noise layer that is seeded with the hash value of the PIDs included in the
respective bucket.

Row counts have two noise layers: one seeded with the hash value of the PIDs present in the bucket and one seeded
with the hash of the bucket's labels and columns (the column names in each tree make up the base seed of the tree and,
for each bucket, the labels are made up from the middle-points of the bucket's ranges).
Then, the regular [Diffix count anonymizer](https://arxiv.org/abs/2201.04351) runs.

## More information

A description of SynDiffix, its privacy properties, and its performance measured over a variety of datasets can be found at
[https://github.com/diffix/syndiffix/wiki/SynDiffix:-Overview](https://github.com/diffix/syndiffix/wiki/SynDiffix:-Overview).
