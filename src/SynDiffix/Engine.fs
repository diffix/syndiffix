module SynDiffix.Engine

open System
open System.IO
open System.Security.Cryptography

open SynDiffix.Clustering
open SynDiffix.Clustering.Clustering
open SynDiffix.Forest
open SynDiffix.Microdata
open SynDiffix.Counter

open SynDiffix.ArgumentParsing

open Thoth.Json.Net

type Result =
  {
    Columns: Column list
    Rows: Row array
    SynRows: Row array
    Forest: Forest
    ElapsedTime: TimeSpan
  }

let private createCountStrategy aidColumns maxThreshold =
  if aidColumns = Csv.rowNumberAid then
    {
      CreateEntityCounter = fun () -> UniqueAidCounter()
      CreateRowCounter = fun () -> UniqueAidCounter()
    }
  else
    {
      CreateEntityCounter = fun () -> GenericAidEntityCounter(aidColumns.Length, maxThreshold)
      CreateRowCounter = fun () -> GenericAidRowCounter(aidColumns.Length)
    }

let private printScores (scores: float[,]) =
  let dims = scores.GetLength(0)

  eprintf "   "

  for i in 0 .. dims - 1 do
    eprintf $"{i.ToString().PadLeft(6, ' ')}"

  eprintfn ""

  for i in 0 .. dims - 1 do
    eprintf $"{i.ToString().PadLeft(2, ' ')} "
    let mutable total = 0.0

    for j in 0 .. dims - 1 do
      if i = j then
        eprintf "  ----"
      else
        total <- total + scores.[i, j]
        eprintf "%s" (scores.[i, j].ToString("0.00").PadLeft(6, ' '))

    eprintfn "  = %s" (total.ToString("0.00"))

  eprintfn ""
  eprintf " = "

  for j in 0 .. dims - 1 do
    let mutable total = 0.0

    for i in 0 .. dims - 1 do
      if i <> j then
        total <- total + scores.[i, j]

    eprintf "%s" (total.ToString("0.00").PadLeft(6, ' '))

  eprintfn ""


let private parseClusters clusters =
  let derivedClustersDecoder =
    Decode.object (fun get ->
      let stitchColumns = get.Required.Field "StitchColumns" (Decode.array Decode.int)
      let derivedColumns = get.Required.Field "DerivedColumns" (Decode.array Decode.int)

      let stitchOwner =
        match (get.Required.Field "StitchOwner" Decode.string).ToLower() with
        | "left" -> StitchOwner.Left
        | "right" -> StitchOwner.Right
        | "shared" -> StitchOwner.Shared
        | _ -> failwith "Invalid stitch owner type!"

      (stitchOwner, stitchColumns, derivedColumns)
    )

  let clustersDecoder =
    Decode.object (fun get ->
      {
        InitialCluster = get.Required.Field "InitialCluster" (Decode.array Decode.int)
        DerivedClusters = get.Required.Field "DerivedClusters" (Decode.list derivedClustersDecoder)
      }
    )

  clusters |> Decode.fromString clustersDecoder |> Result.value

let transform treeCacheLevel (arguments: ParsedArguments) =
  let countStrategy =
    createCountStrategy arguments.AidColumns arguments.BucketizationParams.RangeLowThreshold

  if arguments.Verbose then
    eprintfn "Computing salt..."

  let salt =
    use fileStream = File.Open(arguments.CsvPath, FileMode.Open, FileAccess.Read)
    SHA256.Create().ComputeHash(fileStream)

  let anonParams = { arguments.AnonymizationParams with Salt = salt }

  if arguments.Verbose then
    eprintfn "Reading rows..."

  let rows, columns =
    Csv.read arguments.CsvPath arguments.CsvColumns arguments.AidColumns
    |> mapFst Seq.toArray

  let stopWatch = Diagnostics.Stopwatch.StartNew()

  if arguments.Verbose then
    eprintfn "Building forest..."

  let columnNames, columnTypes =
    columns |> List.map (fun column -> column.Name, column.Type) |> List.unzip

  let dataConvertors = rows |> createDataConvertors columnTypes

  let anonContext = { BucketSeed = Hash.string arguments.CsvPath; AnonymizationParams = anonParams }

  let forest =
    Forest(rows, dataConvertors, anonContext, arguments.BucketizationParams, columnNames, countStrategy)

  forest.SetCacheLevel treeCacheLevel

  let allColumns = [ 0 .. forest.Dimensions - 1 ]

  let clusters: Clustering.Clusters =
    if arguments.BucketizationParams.ClusteringEnabled then
      if arguments.BucketizationParams.ClusteringMaxClusterWeight <= 1 then
        {
          InitialCluster = [| 0 |]
          DerivedClusters =
            allColumns
            |> List.tail
            |> List.map (fun c -> (StitchOwner.Shared, [||], [| c |]))
        }
      elif arguments.Clusters <> "" then
        let clusters =
          if arguments.Clusters.StartsWith("{") then
            arguments.Clusters
          else
            File.ReadAllText arguments.Clusters

        if arguments.Verbose then
          eprintfn $"Manually assigning clusters: %s{clusters}."

        parseClusters clusters
      else
        let forest' = if Sampling.shouldSample forest then Sampling.sampleForest forest else forest

        if arguments.Verbose then
          eprintfn "Measuring dependence..."

        let clusteringContext = Solver.clusteringContext arguments.MainColumn forest'
        // Copy over the measured entropy.
        Array.blit clusteringContext.Entropy1Dim 0 forest.Entropy1Dim 0 forest.Dimensions

        if arguments.Verbose then
          printScores clusteringContext.DependencyMatrix

          eprintfn "=== Columns ==="
          let entropy1Dim = clusteringContext.Entropy1Dim

          columns
          |> List.iteri (fun i col ->
            eprintfn $"{i.ToString().PadLeft(3, ' ')} {col.Name} ({col.Type}); Entropy = {entropy1Dim.[i]}"
          )

        if arguments.Verbose then
          eprintfn "Assigning clusters..."

        Solver.solve clusteringContext
    else
      if arguments.Verbose then
        eprintfn "Using all columns."

      { InitialCluster = List.toArray allColumns; DerivedClusters = [] }

  if arguments.Verbose then
    eprintfn $"Clusters: %A{clusters}."
    eprintfn "Materializing clusters..."

  let synRows, columnIds = Clustering.buildTable materializeTree forest clusters

  if arguments.Verbose then
    eprintfn "Microtable built."

  if Array.toList columnIds <> allColumns then
    failwith "Expected all columns to be present in final microtable."

  stopWatch.Stop()

  if arguments.Verbose then
    eprintfn $"Time elapsed: {stopWatch.Elapsed.TotalMilliseconds} ms."
    eprintfn $"Memory used: {GC.GetTotalMemory(true) / (1024L * 1024L)} MB."

  {
    Columns = columns
    Rows = rows
    SynRows = synRows
    Forest = forest
    ElapsedTime = stopWatch.Elapsed
  }
