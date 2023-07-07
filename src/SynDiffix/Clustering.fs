module SynDiffix.Clustering

open System
open System.Collections.Generic
open FsToolkit.ErrorHandling

open SynDiffix.Combination
open SynDiffix.Forest

// ----------------------------------------------------------------

module Sampling =
  let shouldSample (forest: Forest) =
    let dimensions = forest.Dimensions
    let numRows = forest.Rows.Length
    let numSamples = forest.BucketizationParams.ClusteringTableSampleSize

    if numSamples >= numRows then
      false
    else
      let sampled2dimWork = dimensions * dimensions * numSamples
      let full2dimWork = (numRows * dimensions * 3) / 2

      let totalWorkWithSample = sampled2dimWork + full2dimWork
      let totalWorkWithoutSample = dimensions * dimensions * numRows

      totalWorkWithoutSample > totalWorkWithSample * 2

  let sampleForest (forest: Forest) =
    let samplingAnonContext =
      { forest.AnonymizationContext with
          AnonymizationParams =
            { forest.AnonymizationContext.AnonymizationParams with
                Suppression = { LowThreshold = 2; LowMeanGap = 1; LayerSD = 0.5 }
                LayerNoiseSD = 0.
            }
      }

    // New RNG prevents `forest.Random` from being affected by sample size.
    let random = forest.DeriveUnsafeRandom()

    let origRows = forest.Rows
    let numSamples = forest.BucketizationParams.ClusteringTableSampleSize

    Forest(
      Array.init numSamples (fun _ -> origRows.[random.Next(origRows.Length)]),
      forest.DataConvertors,
      samplingAnonContext,
      forest.BucketizationParams,
      forest.ColumnNames,
      forest.CountStrategy
    )

// ----------------------------------------------------------------

module Dependence =
  // Aliases to help tell what nodes functions expect.
  type private OneDimNode = Tree.Node
  type private TwoDimNode = Tree.Node
  type private AnyDimNode = Tree.Node

  type Score =
    {
      Score: float
      Count: float
      NodeXY: TwoDimNode option
      NodeX: OneDimNode
      NodeY: OneDimNode
    }

  type Result =
    {
      Columns: int * int
      Dependence: float
      Complexity: int64
      Scores: Score list
      MeasureTime: TimeSpan
    }

  let private count (node: AnyDimNode) = node |> Tree.noisyRowCount |> float

  let private getChild index (node: AnyDimNode) =
    match node with
    | Tree.Branch branch -> branch.Children |> Map.tryFind index
    | Tree.Leaf _ -> None

  let private findChild predicate (node: AnyDimNode) =
    match node with
    | Tree.Branch branch ->
      branch.Children
      |> Seq.filter predicate
      |> Seq.toList
      |> function
        | [ child ] -> Some child.Value
        | [] -> None
        | _ -> failwith "Expected to match a single child node."
    | Tree.Leaf _ -> None

  let inline private getBit index (value: int) = (value >>> index) &&& 1

  let private isSingularity (node: OneDimNode) =
    (Tree.nodeData node).ActualRanges.[0].IsSingularity()

  let private singularityValue (node: OneDimNode) =
    (Tree.nodeData node).ActualRanges.[0].Min

  let private snappedRange index (node: OneDimNode) =
    (Tree.nodeData node).SnappedRanges.[index]

  let measureDependence (colX: int) (colY: int) (forest: Forest) : Result =
    // Ensure colX < colY.
    if colX >= colY then
      failwith "Invalid input."

    let numRows = float forest.Rows.Length
    let rangeThresh = forest.BucketizationParams.RangeLowThreshold

    let mustPassRangeThresh (x: float) =
      if x < rangeThresh then None else Some x

    // Walk state
    let scores = MutableList<Score>()
    let mutable complexity = 0L

    let rec walk (nodeXY: TwoDimNode option) (nodeX: OneDimNode) (nodeY: OneDimNode) =
      option {
        // Stop walk if either node has `count < range_thresh`.
        let! countX = nodeX |> count |> mustPassRangeThresh
        let! countY = nodeY |> count |> mustPassRangeThresh

        // Compute and store score.
        let actual2dimCount =
          match nodeXY with
          | Some nodeXY ->
            complexity <- complexity + 1L
            count nodeXY
          | None -> 0.

        let expected2dimCount = (countX * countY) / numRows

        let score =
          (abs (expected2dimCount - actual2dimCount))
          / (max expected2dimCount actual2dimCount)

        scores.Add(
          {
            Score = score
            Count = expected2dimCount
            NodeXY = nodeXY
            NodeX = nodeX
            NodeY = nodeY
          }
        )

        // Walk children.
        // Dim 0 (X) is at bit 1.
        // Dim 1 (Y) is at bit 0.
        if isSingularity nodeX && isSingularity nodeY then
          // Stop walk if both 1-dims are singularities.
          ()
        elif isSingularity nodeX then
          let xSingularity = singularityValue nodeX

          for idChildY in 0..1 do
            nodeY
            |> getChild idChildY
            |> Option.iter (fun childY ->
              // Find child that matches on dim Y. It can be on either side of dim X.
              let childXY =
                nodeXY
                |> Option.bind (
                  findChild (fun pair ->
                    (pair.Key |> getBit 0) = idChildY
                    && (snappedRange 0 pair.Value).ContainsValue(xSingularity)
                  )
                )

              walk childXY nodeX childY
            )
        elif isSingularity nodeY then
          let ySingularity = singularityValue nodeY

          for idChildX in 0..1 do
            nodeX
            |> getChild idChildX
            |> Option.iter (fun childX ->
              // Find child that matches on dim X. It can be on either side of dim Y.
              let childXY =
                nodeXY
                |> Option.bind (
                  findChild (fun pair ->
                    (pair.Key |> getBit 1) = idChildX
                    && (snappedRange 1 pair.Value).ContainsValue(ySingularity)
                  )
                )

              walk childXY childX nodeY
            )
        else
          for idChildXY in 0..3 do
            let idChildX = idChildXY |> getBit 1
            let idChildY = idChildXY |> getBit 0

            option {
              let! childX = nodeX |> getChild idChildX
              let! childY = nodeY |> getChild idChildY
              let childXY = nodeXY |> Option.bind (getChild idChildXY)
              walk childXY childX childY
            }
            |> ignore
      }
      |> ignore

    let rootXY = forest.GetTree([| colX; colY |])
    let rootX = forest.GetTree([| colX |])
    let rootY = forest.GetTree([| colY |])

    let stopwatch = Diagnostics.Stopwatch.StartNew()

    walk (Some rootXY) rootX rootY

    let totalWeightedScore, totalCount =
      if scores.Count > 1 then
        scores
        |> Seq.tail // Skip root measurement.
        |> Seq.fold
          (fun (accScore, accCount) score -> (accScore + score.Score * score.Count, accCount + score.Count))
          (0., 0.)
      else
        0., 0.

    {
      Columns = colX, colY
      Dependence = if totalCount > 0. then totalWeightedScore / totalCount else 0.
      Complexity = complexity
      Scores = Seq.toList scores
      MeasureTime = stopwatch.Elapsed
    }

  type DependenceMeasurements =
    { DependencyMatrix: float[,]; ComplexityMatrix: int64[,]; Entropy1Dim: float[] }

  let private measureEntropy (root: AnyDimNode) =
    let numRows = count root
    let mutable entropy = 0.0

    let rec entropyWalk (node: AnyDimNode) =
      match node with
      | Tree.Branch branch -> branch.Children |> Map.iter (fun _ child -> entropyWalk child)
      | Tree.Leaf _ ->
        let count = count node
        entropy <- entropy - (count / numRows * Math.Log2(count / numRows))

    entropyWalk root
    entropy

  let measureAll (forest: Forest) : DependenceMeasurements =
    let numColumns = forest.Dimensions
    let dependencyMatrix = Array2D.create<float> numColumns numColumns 1.0
    let complexityMatrix = Array2D.create<int64> numColumns numColumns 0
    let entropy1Dim = Array.init numColumns (fun i -> forest.GetTree([| i |]) |> measureEntropy)

    generateCombinations 2 numColumns
    |> Seq.iter (fun comb ->
      let x, y = comb[0], comb[1]
      let score = (forest |> measureDependence x y)

      let dependence = score.Dependence
      dependencyMatrix.[x, y] <- dependence
      dependencyMatrix.[y, x] <- dependence
      complexityMatrix.[x, y] <- score.Complexity
      complexityMatrix.[y, x] <- score.Complexity
    )

    {
      DependencyMatrix = dependencyMatrix
      ComplexityMatrix = complexityMatrix
      Entropy1Dim = entropy1Dim
    }

// ----------------------------------------------------------------

module Clustering =
  type ColumnId = int // Global index of a column.
  type private ColumnIndex = int // Local index of a column in a micro-table.

  type StitchOwner =
    | Left = 0
    | Right = 1
    | Shared = 2

  type DerivedCluster = StitchOwner * ColumnId array * ColumnId array // owner, stitch columns, derived columns

  type Clusters = { InitialCluster: ColumnId array; DerivedClusters: DerivedCluster list }

  type TreeMaterializer = Forest -> ColumnId seq -> MicrodataRow array * Combination

  let private isIntegral (dataConvertor: IDataConvertor) =
    match dataConvertor.ColumnType with
    | RealType
    | TimestampType -> false
    | _ -> true

  let private shuffleRows (random: Random) (rows: Span<MicrodataRow>) =
    let mutable i = rows.Length - 1

    while i > 0 do
      let j = random.Next(i + 1)
      let temp = rows.[i]
      rows.[i] <- rows.[j]
      rows.[j] <- temp
      i <- i - 1

  let private averageLength microTables =
    microTables |> List.averageBy (fst >> Array.length >> float) |> round |> int

  let private alignLength (random: Random) length (microTable: Span<MicrodataRow>) : Span<MicrodataRow> =
    if length = microTable.Length then
      microTable
    elif length < microTable.Length then
      microTable.Slice(0, length)
    else
      let copy = Array.zeroCreate length
      microTable.CopyTo(Span(copy))

      for i = microTable.Length to length - 1 do
        copy.[i] <- microTable[random.Next(microTable.Length)]

      Span(copy)

  let private findIndexes subset superset =
    subset |> Array.map (fun c -> Array.findIndex ((=) c) superset)

  type private ColumnLocation =
    {
      SourceRow: StitchOwner
      LeftIndex: ColumnIndex
      RightIndex: ColumnIndex
      ColumnId: ColumnId
    }

  let private locateColumns (leftCombination: Combination) (rightCombination: Combination) : ColumnLocation array =
    [ leftCombination; rightCombination ]
    |> Seq.indexed
    |> Seq.collect (fun (tableIndex, columns) ->
      columns
      |> Array.mapi (fun columnIndex columnId -> (tableIndex, columnIndex, columnId))
    )
    |> Seq.groupBy thd3
    |> Seq.map (fun (columnId, sources) ->
      match Seq.toList sources with
      | [] -> failwith "Impossible."
      | [ (tableIndex, columnIndex, columnId) ] ->
        {
          SourceRow = if tableIndex = 0 then StitchOwner.Left else StitchOwner.Right
          LeftIndex = if tableIndex = 0 then columnIndex else -1
          RightIndex = if tableIndex = 1 then columnIndex else -1
          ColumnId = columnId
        }
      | bothSides ->
        let sorted = bothSides |> List.sortBy fst3

        {
          SourceRow = StitchOwner.Shared
          LeftIndex = sorted.[0] |> snd3
          RightIndex = sorted.[1] |> snd3
          ColumnId = columnId
        }
    )
    |> Seq.sortBy (fun c -> c.ColumnId)
    |> Seq.toArray

  let rec private binarySearch
    (rows: Span<MicrodataRow>)
    (column: ColumnIndex)
    (target: float)
    (start: int)
    (endExclusive: int)
    =
    if start >= endExclusive then
      -1
    else
      let mid = (start + endExclusive) / 2

      if vsnd rows.[mid].[column] >= target then
        if mid = 0 || vsnd rows.[mid - 1].[column] < target then
          mid
        else
          binarySearch rows column target start mid
      else
        binarySearch rows column target (mid + 1) endExclusive

  type private SingleColumnComparer(key: ColumnIndex) =
    interface IComparer<MicrodataRow> with
      member this.Compare(x, y) = (vsnd x.[key]).CompareTo(vsnd y.[key])

  type private MultipleColumnsComparer(keys: ColumnIndex array) =
    interface IComparer<MicrodataRow> with
      member this.Compare(x, y) =
        let mutable result = 0
        let mutable i = 0

        while result = 0 && i < keys.Length do
          let key = keys.[i]
          result <- (vsnd x.[key]).CompareTo(vsnd y.[key])
          i <- i + 1

        result

  let private makeComparer (keys: ColumnIndex array) : IComparer<MicrodataRow> =
    if keys.Length = 1 then
      SingleColumnComparer(keys.[0])
    else
      MultipleColumnsComparer(keys)

  type private StitchContext =
    {
      Random: Random
      StitchOwner: StitchOwner
      AllColumns: ColumnLocation array
      Entropy1Dim: float[]
      StitchMaxValues: float[]
      StitchIsIntegral: bool[]
      LeftStitchIndexes: ColumnIndex array
      RightStitchIndexes: ColumnIndex array
      ResultRows: MutableList<MicrodataRow>
    }
    member this.NumStitchColumns = this.LeftStitchIndexes.Length

  type private StitchState =
    {
      Depth: int
      StitchRanges: Range.Ranges
      NextSortColumn: int
      CurrentlySortedBy: int
      RemainingSortAttempts: int
      Context: StitchContext
    }

  let private mergeRow
    (allColumns: ColumnLocation array)
    (pickSharedLeft: bool)
    (leftRow: MicrodataRow)
    (rightRow: MicrodataRow)
    =
    let numCols = allColumns.Length
    let mergedRow = Array.zeroCreate numCols

    for j = 0 to numCols - 1 do
      let col = allColumns.[j]

      if col.SourceRow = StitchOwner.Left then
        mergedRow.[j] <- leftRow.[col.LeftIndex]
      elif col.SourceRow = StitchOwner.Right then
        mergedRow.[j] <- rightRow.[col.RightIndex]
      elif pickSharedLeft then
        mergedRow.[j] <- leftRow.[col.LeftIndex]
      else
        mergedRow.[j] <- rightRow.[col.RightIndex]

    mergedRow

  let private mergeMicrodata (state: StitchState) (leftRows: Span<MicrodataRow>) (rightRows: Span<MicrodataRow>) =
    if leftRows.Length = 0 || rightRows.Length = 0 then
      failwith "Attempted a stitch with no rows."

    let context = state.Context
    let stitchOwner = context.StitchOwner
    let random = context.Random
    let allColumns = context.AllColumns

    let numRows =
      match stitchOwner with
      | StitchOwner.Left -> leftRows.Length
      | StitchOwner.Right -> rightRows.Length
      | _ -> ((float leftRows.Length + float rightRows.Length) / 2.0) |> round |> int

    // This time we shuffle to remove previous sorting in order to safely discard rows.

    // Process one side at a time for hopefully better CPU caching.
    shuffleRows random leftRows
    let leftRows = alignLength random numRows leftRows
    leftRows.Sort(makeComparer context.LeftStitchIndexes)

    shuffleRows random rightRows
    let rightRows = alignLength random numRows rightRows
    rightRows.Sort(makeComparer context.RightStitchIndexes)

    for i = 0 to numRows - 1 do
      let pickSharedLeft =
        if stitchOwner = StitchOwner.Shared then
          i % 2 = 0
        else
          stitchOwner = StitchOwner.Left

      context.ResultRows.Add(mergeRow allColumns pickSharedLeft leftRows.[i] rightRows.[i])

  [<Literal>]
  let private THRESH_REL = 0.7

  let private acceptableDistribution (left: int) (right: int) =
    let min = min left right
    let max = max left right

    if min = 0 then
      false
    else
      let relDiff = float min / float max
      relDiff >= THRESH_REL

  let private canSplit (state: StitchState) =
    let context = state.Context
    let splitColumn = state.NextSortColumn

    if context.StitchIsIntegral.[splitColumn] then
      let range = state.StitchRanges.[splitColumn]

      if context.StitchMaxValues.[splitColumn] = range.Max then
        range.Size() >= 1.0
      else
        range.Size() > 1.0
    else
      true

  let rec private stitchRec (state: StitchState) (leftRows: Span<MicrodataRow>) (rightRows: Span<MicrodataRow>) =
    if state.RemainingSortAttempts = 0 || leftRows.Length = 1 || rightRows.Length = 1 then
      mergeMicrodata state leftRows rightRows
    elif canSplit state then
      let context = state.Context
      let currentSortColumn = state.NextSortColumn

      let leftStitchIndex = context.LeftStitchIndexes.[currentSortColumn]
      let rightStitchIndex = context.RightStitchIndexes.[currentSortColumn]

      if state.CurrentlySortedBy <> currentSortColumn then
        leftRows.Sort(SingleColumnComparer(leftStitchIndex))
        rightRows.Sort(SingleColumnComparer(rightStitchIndex))

      let range = state.StitchRanges.[currentSortColumn]
      let rangeMiddle = range.Middle()

      let leftSplitPoint = binarySearch leftRows leftStitchIndex rangeMiddle 0 leftRows.Length |> max 0
      let rightSplitPoint = binarySearch rightRows rightStitchIndex rangeMiddle 0 rightRows.Length |> max 0

      let leftLower = leftRows.Slice(0, leftSplitPoint)
      let rightLower = rightRows.Slice(0, rightSplitPoint)

      let leftUpper = leftRows.Slice(leftSplitPoint)
      let rightUpper = rightRows.Slice(rightSplitPoint)

      if
        acceptableDistribution leftLower.Length rightLower.Length
        && acceptableDistribution leftUpper.Length rightUpper.Length
      then
        // Visit lower half.
        stitchRec
          { state with
              Depth = state.Depth + 1
              StitchRanges = state.StitchRanges |> Array.updateAt currentSortColumn (range.LowerHalf())
              NextSortColumn = (currentSortColumn + 1) % context.NumStitchColumns
              CurrentlySortedBy = currentSortColumn
              RemainingSortAttempts = context.NumStitchColumns
          }
          leftLower
          rightLower

        // Visit upper half.
        stitchRec
          { state with
              Depth = state.Depth + 1
              StitchRanges = state.StitchRanges |> Array.updateAt currentSortColumn (range.UpperHalf())
              NextSortColumn = (currentSortColumn + 1) % context.NumStitchColumns
              CurrentlySortedBy = currentSortColumn
              RemainingSortAttempts = context.NumStitchColumns
          }
          leftUpper
          rightUpper
      else
        let nextStitchRanges =
          if leftLower.Length = 0 && rightLower.Length = 0 then
            state.StitchRanges |> Array.updateAt currentSortColumn (range.UpperHalf())
          elif leftUpper.Length = 0 && rightUpper.Length = 0 then
            state.StitchRanges |> Array.updateAt currentSortColumn (range.LowerHalf())
          else
            state.StitchRanges

        // Try next column.
        stitchRec
          { state with
              NextSortColumn = (currentSortColumn + 1) % context.NumStitchColumns
              StitchRanges = nextStitchRanges
              CurrentlySortedBy = currentSortColumn
              RemainingSortAttempts = state.RemainingSortAttempts - 1
          }
          leftRows
          rightRows
    else
      // Try next column.
      stitchRec
        { state with
            NextSortColumn = (state.NextSortColumn + 1) % state.Context.NumStitchColumns
            RemainingSortAttempts = state.RemainingSortAttempts - 1
        }
        leftRows
        rightRows

  let private doStitch
    (forest: Forest)
    ((leftRows, leftCombination): MicrodataRow array * Combination)
    ((rightRows, rightCombination): MicrodataRow array * Combination)
    ((stitchOwner, stitchColumns, derivedColumns): DerivedCluster)
    : MicrodataRow array * Combination =
    if
      leftCombination.Length = 0
      || stitchColumns.Length = 0
      || derivedColumns.Length = 0
    then
      failwith "Invalid clusters in stitch operation."

    if rightRows.Length = 0 then
      failwith $"Empty sequence in cluster %A{rightCombination}."

    // Pick lowest entropy column first.
    let stitchColumns = stitchColumns |> Array.sortBy (fun col -> forest.Entropy1Dim.[col], col)
    let allColumns = locateColumns leftCombination rightCombination
    let resultRows = MutableList<MicrodataRow>(leftRows.Length)

    let rootStitchRanges = forest.SnappedRanges |> getItemsCombination stitchColumns

    let stitchState =
      {
        Depth = 0
        StitchRanges = rootStitchRanges
        NextSortColumn = 0
        CurrentlySortedBy = -1
        RemainingSortAttempts = stitchColumns.Length
        Context =
          {
            Random = forest.Random
            StitchOwner = stitchOwner
            AllColumns = allColumns
            Entropy1Dim = forest.Entropy1Dim |> getItemsCombination stitchColumns
            StitchMaxValues = rootStitchRanges |> Array.map (fun r -> r.Max)
            StitchIsIntegral =
              forest.DataConvertors
              |> getItemsCombination stitchColumns
              |> Array.map isIntegral
            LeftStitchIndexes = leftCombination |> findIndexes stitchColumns
            RightStitchIndexes = rightCombination |> findIndexes stitchColumns
            ResultRows = resultRows
          }
      }

    stitchRec stitchState (Span(leftRows)) (Span(rightRows))

    Seq.toArray resultRows, allColumns |> Array.map (fun c -> c.ColumnId)

  let private doPatch
    (random: Random)
    ((leftRows, leftCombination): MicrodataRow array * Combination)
    ((rightRows, rightCombination): MicrodataRow array * Combination)
    : MicrodataRow array * Combination =
    let DOESNT_MATTER = true

    let allColumns = locateColumns leftCombination rightCombination
    let numRows = leftRows.Length

    let leftRows = Span(leftRows)
    let rightRows = Span(rightRows)

    // No need to shuffle left rows in a patch.

    // F# does not allow piping ref structs...
    shuffleRows random rightRows
    let rightRows = alignLength random numRows rightRows

    let allRows = Array.zeroCreate<MicrodataRow> numRows

    for i = 0 to numRows - 1 do
      allRows.[i] <- mergeRow allColumns DOESNT_MATTER leftRows.[i] rightRows.[i]

    allRows, allColumns |> Array.map (fun c -> c.ColumnId)

  let private stitch
    (materializeTree: TreeMaterializer)
    (forest: Forest)
    (left: MicrodataRow array * Combination)
    ((stitchOwner, stitchColumns, derivedColumns): DerivedCluster)
    : MicrodataRow array * Combination =
    let right = materializeTree forest (Array.append stitchColumns derivedColumns)

    if stitchColumns.Length = 0 then
      doPatch forest.Random left right
    else
      doStitch forest left right (stitchOwner, stitchColumns, derivedColumns)

  let buildTable (materializeTree: TreeMaterializer) (forest: Forest) (clusters: Clusters) =
    clusters.DerivedClusters
    |> List.fold (stitch materializeTree forest) (materializeTree forest clusters.InitialCluster)
    |> mapFst (Array.map microdataRowToRow)

// ----------------------------------------------------------------

open Clustering

module Solver =
  type private JoinTree = { JoinColumn: ColumnId; Children: JoinTree list }

  type ClusteringContext =
    {
      DependencyMatrix: float[,]
      ComplexityMatrix: int64[,]
      Entropy1Dim: float[]
      TotalDependencePerColumn: float[]
      AnonymizationParams: AnonymizationParams
      BucketizationParams: BucketizationParams
      Random: Random
    }
    member this.NumColumns = this.DependencyMatrix.GetLength(0)

  type private MutableCluster = { Columns: HashSet<ColumnId>; mutable TotalEntropy: float }

  [<Literal>]
  let private DERIVED_COLS_MIN = 1

  [<Literal>]
  let private DERIVED_COLS_RESERVED = 0.5

  [<Literal>]
  let private DERIVED_COLS_RATIO = 0.7

  let private buildClusters (context: ClusteringContext) (permutation: int array) : Clusters =
    let mergeThresh = context.BucketizationParams.ClusteringMergeThreshold
    let maxWeight = context.BucketizationParams.ClusteringMaxClusterWeight

    let dependencyMatrix = context.DependencyMatrix
    let entropy1Dim = context.Entropy1Dim

    let clusters = MutableList<MutableCluster>()

    let colWeight col = 1.0 + sqrt (max entropy1Dim.[col] 1.0)

    // For each column in the permutation, we find the "best" cluster that has available space.
    // We judge how suitable a cluster is by average dependence score.
    // If no available cluster is found, we start a new one.
    // After all clusters are built, we fill remaining space of non-initial clusters with stitch columns.

    for col in permutation do
      let bestCluster =
        clusters
        |> Seq.mapi (fun i cluster ->
          cluster,
          (if i = 0 then maxWeight else DERIVED_COLS_RATIO * maxWeight),
          cluster.Columns |> Seq.averageBy (fun c -> dependencyMatrix.[col, c])
        )
        |> Seq.filter (fun (cluster, capacity, averageQuality) ->
          averageQuality >= mergeThresh
          && (cluster.Columns.Count < DERIVED_COLS_MIN
              || cluster.TotalEntropy + colWeight col <= capacity)
        )
        |> Seq.sortByDescending thd3
        |> Seq.tryHead

      match bestCluster with
      | Some(bestCluster, _capacity, _averageQuality) ->
        bestCluster.Columns.Add(col) |> ignore
        bestCluster.TotalEntropy <- bestCluster.TotalEntropy + colWeight col
      | None -> clusters.Add({ Columns = HashSet([ col ]); TotalEntropy = colWeight col })

    let derivedClusters = MutableList<DerivedCluster>()
    let availableColumns = HashSet<ColumnId>(clusters.[0].Columns)

    for i = 1 to clusters.Count - 1 do
      let cluster = clusters.[i]
      let mutable totalWeight = max cluster.TotalEntropy (DERIVED_COLS_RESERVED * maxWeight)

      let stitchColumns = HashSet<ColumnId>()
      let derivedColumns = Seq.toArray cluster.Columns

      let bestStitchColumns =
        availableColumns
        |> Seq.map (fun cLeft ->
          let mutable depSum = 0.0
          let mutable depMax = -1.0

          for cRight in derivedColumns do
            let dep = dependencyMatrix.[cLeft, cRight]
            depSum <- depSum + dep
            depMax <- max depMax dep

          let depAvg = depSum / float derivedColumns.Length

          cLeft, depAvg, depMax
        )
        |> Seq.sortByDescending (fun (cLeft, depAvg, depMax) ->
          depAvg |> Math.floorBy 0.05, depMax |> Math.floorBy 0.01, context.TotalDependencePerColumn.[cLeft]
        )
        |> Seq.toArray

      // Always pick best match.
      do
        let bestStitchCol = fst3 bestStitchColumns.[0]
        stitchColumns.Add(bestStitchCol) |> ignore
        totalWeight <- totalWeight + colWeight bestStitchCol

      // Add remaining matches, as many as possible.
      for i = 1 to bestStitchColumns.Length - 1 do
        let cLeft, _depAvg, depMax = bestStitchColumns.[i]

        if depMax >= mergeThresh then
          let weight = colWeight cLeft

          if totalWeight + weight <= maxWeight then
            stitchColumns.Add(cLeft) |> ignore
            totalWeight <- totalWeight + weight

      availableColumns.UnionWith(derivedColumns)
      derivedClusters.Add((StitchOwner.Shared, Seq.toArray stitchColumns, derivedColumns))

    {
      InitialCluster = Seq.toArray clusters.[0].Columns
      DerivedClusters = Seq.toList derivedClusters
    }

  let private clusteringQuality (context: ClusteringContext) (clusters: Clusters) =
    let dependencyMatrix = context.DependencyMatrix

    let unsatisfiedDependencies = Array.copy context.TotalDependencePerColumn

    let visitPairs (columns: ColumnId array) =
      for i = 1 to columns.Length - 1 do
        let colA = columns.[i]

        for j = 0 to i - 1 do
          let colB = columns.[j]
          let dependence = dependencyMatrix.[colA, colB] // Assumes a symmetric matrix.
          unsatisfiedDependencies.[colA] <- unsatisfiedDependencies.[colA] - dependence
          unsatisfiedDependencies.[colB] <- unsatisfiedDependencies.[colB] - dependence

    visitPairs clusters.InitialCluster

    for _, stitchColumns, derivedColumns in clusters.DerivedClusters do
      visitPairs (Array.append stitchColumns derivedColumns)

    Array.sum unsatisfiedDependencies / (2.0 * float unsatisfiedDependencies.Length)


  let clusteringContext (mainColumn: ColumnId option) (forest: Forest) =
    let measures = Dependence.measureAll forest
    let dependencyMatrix = measures.DependencyMatrix

    if mainColumn.IsSome then
      let mainColumn = mainColumn.Value

      for i = 0 to forest.Dimensions - 1 do
        dependencyMatrix.[i, mainColumn] <- dependencyMatrix.[i, mainColumn] + 0.5
        dependencyMatrix.[mainColumn, i] <- dependencyMatrix.[mainColumn, i] + 0.5

    let totalPerColumn = Array.zeroCreate<float> forest.Dimensions

    for i = 0 to forest.Dimensions - 1 do
      for j = 0 to forest.Dimensions - 1 do
        if i <> j then
          totalPerColumn.[i] <- totalPerColumn.[i] + dependencyMatrix.[i, j]

    {
      DependencyMatrix = dependencyMatrix
      ComplexityMatrix = measures.ComplexityMatrix
      Entropy1Dim = measures.Entropy1Dim
      TotalDependencePerColumn = totalPerColumn
      AnonymizationParams = forest.AnonymizationContext.AnonymizationParams
      BucketizationParams = forest.BucketizationParams
      Random = forest.Random
    }

  let private doSolve (context: ClusteringContext) =
    let numCols = context.NumColumns
    let random = context.Random

    // Constants
    let initialSolution = [| 0 .. numCols - 1 |]

    let initialTemperature = 5.0
    let minTemperature = 3.5E-3
    let alpha = 1.5E-3

    let nextTemperature currentTemp =
      currentTemp / (1.0 + alpha * currentTemp)

    let mutate (solution: int array) =
      let copy = Array.copy solution
      let i = random.Next(numCols)
      let mutable j = random.Next(numCols)

      while i = j do
        j <- random.Next(numCols)

      copy.[i] <- solution.[j]
      copy.[j] <- solution.[i]
      copy

    let evaluate (solution: int array) =
      let clusters = buildClusters context solution
      clusteringQuality context clusters

    // Solver state
    let mutable currentSolution = initialSolution
    let mutable currentEnergy = evaluate initialSolution
    let mutable bestSolution = initialSolution
    let mutable bestEnergy = currentEnergy
    let mutable temperature = initialTemperature

    // Simulated annealing loop
    while bestEnergy > 0 && temperature > minTemperature do
      let newSolution = mutate currentSolution
      let newEnergy = evaluate newSolution
      let energyDelta = newEnergy - currentEnergy

      if energyDelta <= 0.0 || Math.Exp(-energyDelta / temperature) > random.NextDouble() then
        currentSolution <- newSolution
        currentEnergy <- newEnergy

      if currentEnergy < bestEnergy then
        bestSolution <- currentSolution
        bestEnergy <- currentEnergy

      temperature <- nextTemperature temperature

    buildClusters context bestSolution

  let solve (context: ClusteringContext) =
    assert (context.BucketizationParams.ClusteringMaxClusterWeight > 1)
    let numCols = context.NumColumns

    if numCols >= 3 then
      // TODO: Do an exact search up to a number of columns.
      doSolve context
    else
      // Build a cluster that includes everything.
      { InitialCluster = [| 0 .. numCols - 1 |]; DerivedClusters = [] }
