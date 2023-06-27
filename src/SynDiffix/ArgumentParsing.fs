module SynDiffix.ArgumentParsing

open Argu

type private Arguments =
  | [<MainCommand; ExactlyOnce; First>] CsvPath of string
  | [<ExactlyOnce>] Columns of string list
  | [<Unique>] AidColumns of string list
  | [<Unique; AltCommandLine("-lcf")>] Lcf_Low_Threshold of int
  | [<Unique; AltCommandLine("-range")>] Range_Low_Threshold of int
  | [<Unique; AltCommandLine("-sing")>] Singularity_Low_Threshold of int
  | [<Unique; AltCommandLine("-thresh_sd")>] Threshold_SD of float
  | [<Unique>] Outlier_Count of int * int
  | [<Unique>] Top_Count of int * int
  | [<Unique; AltCommandLine("-noise_sd")>] Layer_Noise_SD of float
  | No_Clustering
  | [<Unique>] Clusters of string
  | [<Unique>] Clustering_MainColumn of string
  | [<Unique>] Clustering_SampleSize of int
  | [<Unique>] Clustering_MaxWeight of float
  | [<Unique>] Clustering_Thresh_Merge of float
  | [<Unique>] Precision_Limit_Row_Fraction of int
  | [<Unique>] Precision_Limit_Depth_Threshold of int
  | Verbose
  | Debug

  interface IArgParserTemplate with
    member this.Usage =
      match this with
      | CsvPath _ -> "Path to the CSV file."
      | Columns _ ->
        "List of columns and their types in the format `column:t`, where `t` is one of the following: "
        + "`b`-boolean, `i`-integer, `r`-real, `t`-timestamp, `s`-string."
      | AidColumns _ ->
        "List of entity ID columns used for anonymization. If not specified, "
        + "assumes each row represents a different entity."
      | Lcf_Low_Threshold _ -> "Low threshold for the low-count filter."
      | Range_Low_Threshold _ -> "Low threshold for a range bucket."
      | Singularity_Low_Threshold _ -> "Low threshold for a singularity bucket."
      | Threshold_SD _ -> "Threshold SD for LCF/range/singularity decisions."
      | Outlier_Count _ -> "Outlier count interval."
      | Top_Count _ -> "Top values count interval."
      | Layer_Noise_SD _ -> "Count SD for each noise layer."
      | No_Clustering -> "Disables column clustering."
      | Clustering_MainColumn _ -> "Column to be prioritized in clusters."
      | Clusters _ -> "The JSON file defining the clusters."
      | Clustering_SampleSize _ -> "Table sample size when measuring dependence."
      | Clustering_MaxWeight _ -> "Maximum cluster size, in weight units."
      | Clustering_Thresh_Merge _ -> "Dependence threshold for combining columns in a cluster."
      | Precision_Limit_Row_Fraction _ ->
        "Tree nodes are allowed to split if `node_num_rows >= table_num_rows/row_fraction`."
      | Precision_Limit_Depth_Threshold _ -> "Tree depth threshold below which the `row-fraction` check is not applied."
      | Verbose -> "Log extra information to stderr."
      | Debug -> "Display extra output for debugging purposes."

type ParsedArguments =
  {
    CsvPath: string
    CsvColumns: Column list
    AidColumns: string list
    AnonymizationParams: AnonymizationParams
    BucketizationParams: BucketizationParams
    MainColumn: int option
    Verbose: bool
    Debug: bool
    Clusters: string
  }

let private parseColumnReference (str: string) =
  let columnType =
    Csv.columnTypeFromName str
    |> Option.defaultWith (fun _ -> failwith $"Cannot determine type for column `{str}`.")

  let name = str.Substring(0, str.LastIndexOf(":")) // Discards type annotation when it is provided from CLI.
  { Name = name; Type = columnType }

let private setLcfLowThreshold (parsedArguments: ParseResults<Arguments>) anonParams =
  match parsedArguments.TryGetResult Lcf_Low_Threshold with
  | Some threshold ->
    assert (threshold >= 2)

    { anonParams with
        Suppression = { anonParams.Suppression with LowThreshold = threshold }
    }
  | None -> anonParams

let private setRangeLowThreshold (parsedArguments: ParseResults<Arguments>) bucketizationParams =
  match parsedArguments.TryGetResult Range_Low_Threshold with
  | Some threshold ->
    assert (threshold >= 2)
    { bucketizationParams with RangeLowThreshold = threshold }
  | None -> bucketizationParams

let private setSingularityLowThreshold (parsedArguments: ParseResults<Arguments>) bucketizationParams =
  match parsedArguments.TryGetResult Singularity_Low_Threshold with
  | Some threshold ->
    assert (threshold >= 2)
    { bucketizationParams with SingularityLowThreshold = threshold }
  | None -> bucketizationParams

let private setThresholdSD (parsedArguments: ParseResults<Arguments>) anonParams =
  match parsedArguments.TryGetResult Threshold_SD with
  | Some sd ->
    assert (sd >= 0.0)
    { anonParams with Suppression = { anonParams.Suppression with LayerSD = sd } }
  | None -> anonParams

let private setOutlierCount (parsedArguments: ParseResults<Arguments>) anonParams =
  match parsedArguments.TryGetResult Outlier_Count with
  | Some(min, max) ->
    assert (min >= 0)
    assert (max >= min)
    { anonParams with OutlierCount = { Lower = min; Upper = max } }
  | None -> anonParams

let private setTopCount (parsedArguments: ParseResults<Arguments>) anonParams =
  match parsedArguments.TryGetResult Top_Count with
  | Some(min, max) ->
    assert (min >= 0)
    assert (max >= min)
    { anonParams with TopCount = { Lower = min; Upper = max } }
  | None -> anonParams

let private setLayerNoiseSD (parsedArguments: ParseResults<Arguments>) anonParams =
  match parsedArguments.TryGetResult Layer_Noise_SD with
  | Some sd ->
    assert (sd >= 0.0)
    { anonParams with LayerNoiseSD = sd }
  | None -> anonParams

let private setClusteringParams (parsedArguments: ParseResults<Arguments>) bucketizationParams =
  { bucketizationParams with
      ClusteringEnabled =
        if (parsedArguments.TryGetResult No_Clustering).IsSome then
          false
        else
          bucketizationParams.ClusteringEnabled
      ClusteringTableSampleSize =
        parsedArguments.TryGetResult Clustering_SampleSize
        |> Option.defaultValue bucketizationParams.ClusteringTableSampleSize
      ClusteringMaxClusterWeight =
        parsedArguments.TryGetResult Clustering_MaxWeight
        |> Option.defaultValue bucketizationParams.ClusteringMaxClusterWeight
      ClusteringMergeThreshold =
        parsedArguments.TryGetResult Clustering_Thresh_Merge
        |> Option.defaultValue bucketizationParams.ClusteringMergeThreshold
  }

let private setPrecisionLimit (parsedArguments: ParseResults<Arguments>) bucketizationParams =
  { bucketizationParams with
      PrecisionLimitDepthThreshold =
        parsedArguments.TryGetResult Precision_Limit_Depth_Threshold
        |> Option.defaultValue bucketizationParams.PrecisionLimitDepthThreshold
      PrecisionLimitRowFraction =
        parsedArguments.TryGetResult Precision_Limit_Row_Fraction
        |> Option.defaultValue bucketizationParams.PrecisionLimitRowFraction
  }

let parseArguments argv =
  let parser = ArgumentParser.Create<Arguments>()

  let parsedArguments =
    parser.ParseCommandLine(inputs = argv, raiseOnUsage = true, ignoreMissing = false, ignoreUnrecognized = false)

  let csvPath = parsedArguments.GetResult CsvPath

  let verbose = (parsedArguments.TryGetResult Verbose).IsSome
  let debug = (parsedArguments.TryGetResult Debug).IsSome

  let clusters = parsedArguments.TryGetResult Clusters |> Option.defaultValue ""

  let columns = parsedArguments.TryGetResult Columns
  let csvColumns = columns.Value |> List.map parseColumnReference

  let aidColumns = parsedArguments.TryGetResult AidColumns |> Option.defaultValue Csv.rowNumberAid

  let anonParams =
    AnonymizationParams.Default
    |> setLcfLowThreshold parsedArguments
    |> setThresholdSD parsedArguments
    |> setOutlierCount parsedArguments
    |> setTopCount parsedArguments
    |> setLayerNoiseSD parsedArguments

  let bucketizationParams =
    BucketizationParams.Default
    |> setRangeLowThreshold parsedArguments
    |> setSingularityLowThreshold parsedArguments
    |> setClusteringParams parsedArguments
    |> setPrecisionLimit parsedArguments

  let mainColumn =
    parsedArguments.TryGetResult Clustering_MainColumn
    |> Option.map (fun colName -> csvColumns |> List.findIndex (fun c -> c.Name = colName))

  {
    CsvPath = csvPath
    CsvColumns = csvColumns
    AidColumns = aidColumns
    AnonymizationParams = anonParams
    BucketizationParams = bucketizationParams
    MainColumn = mainColumn
    Verbose = verbose
    Debug = debug
    Clusters = clusters
  }
