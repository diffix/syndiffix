[<AutoOpen>]
module SynDiffix.Common

open System
open System.Globalization

type Value =
  | Null
  | Boolean of bool
  | Integer of int64
  | Real of float
  | String of string
  | Timestamp of DateTime
  | List of Value list

module Value =
  /// Converts a value to its string representation.
  let rec toString value =
    match value with
    | Null -> "NULL"
    | Boolean true -> "t"
    | Boolean false -> "f"
    | Integer i -> i.ToString()
    | Real r -> r.ToString()
    | String s -> s
    | Timestamp ts -> ts.ToString("o")
    | List values -> "[" + (values |> List.map toStringListItem |> String.joinWithComma) + "]"

  and private toStringListItem value =
    match value with
    | String s -> String.quoteSingle s
    | value -> toString value

  let addToSeed seed (values: Value seq) =
    values |> Seq.map toString |> Hash.strings seed

type AidHash = Hash

type Row = Value array

type ExpressionType =
  | BooleanType
  | IntegerType
  | RealType
  | StringType
  | TimestampType
  | ListType of ExpressionType
  | UnknownType of string

type Column = { Name: string; Type: ExpressionType }

type Columns = Column list

// ----------------------------------------------------------------
// Value utils
// ----------------------------------------------------------------

type OrderByDirection =
  | Ascending
  | Descending

type OrderByNullsBehavior =
  | NullsFirst
  | NullsLast

/// Returns a value comparer with given direction and nulls behavior.
let comparer direction nulls =
  let directionValue =
    match direction with
    | Ascending -> 1
    | Descending -> -1

  let nullsValue =
    match nulls with
    | NullsFirst -> -1
    | NullsLast -> 1

  fun a b ->
    match a, b with
    | Null, Null -> 0
    | Null, _ -> nullsValue
    | _, Null -> -nullsValue
    | String x, String y ->
      // Using PostgreSQL string comparison as a template.
      // https://wiki.postgresql.org/wiki/FAQ#Why_do_my_strings_sort_incorrectly.3F

      // We want whitespace & punctuation comparison ("symbols" in .NET) to have smaller priority,
      // so we ignore them first.
      let comparisonIgnoreSymbols =
        directionValue
        * CultureInfo.InvariantCulture.CompareInfo.Compare(x, y, CompareOptions.IgnoreSymbols)
      // If the former gives a tie, we include symbols.
      if (comparisonIgnoreSymbols <> 0) then
        comparisonIgnoreSymbols
      else
        directionValue * Operators.compare x y
    | x, y -> directionValue * Operators.compare x y

let MONEY_ROUND_MIN = 1e-10
let MONEY_ROUND_DELTA = MONEY_ROUND_MIN / 100.0

// Works with `value` between 1.0 and 10.0.
let private moneyRoundInternal value =
  if value >= 1.0 && value < 1.5 then 1.0
  else if value >= 1.5 && value < 3.5 then 2.0
  else if value >= 3.5 && value < 7.5 then 5.0
  else 10.0

let moneyRound value =
  if value >= 0.0 && value < MONEY_ROUND_MIN then
    0.0
  else
    let tens = Math.Pow(10.0, floor (Math.Log10(value)))
    tens * (moneyRoundInternal (value / tens))

let isMoneyRounded arg =
  match arg with
  // "money-style" numbers, i.e. 1, 2, or 5 preceded by or followed by zeros: ⟨... 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, ...⟩
  | Real c -> abs (moneyRound (c) - c) < MONEY_ROUND_DELTA
  | Integer c -> abs (moneyRound (float c) - float c) < MONEY_ROUND_DELTA
  | _ -> false

// ----------------------------------------------------------------
// Anonymizer types
// ----------------------------------------------------------------

type Interval =
  {
    Lower: int
    Upper: int
  }
  static member Default = { Lower = 2; Upper = 5 }

type TableSettings = { AidColumns: string list }

type LowCountParams = { LowThreshold: int; LayerSD: float; LowMeanGap: float }

type SuppressionParams =
  static member Default = { LowThreshold = 3; LayerSD = 1.; LowMeanGap = 2. }

type BucketizationParams =
  {
    SingularityLowThreshold: int
    RangeLowThreshold: int
    ClusteringEnabled: bool
    ClusteringTableSampleSize: int
    ClusteringMaxClusterWeight: float
    ClusteringMergeThreshold: float
    PrecisionLimitRowFraction: int
    PrecisionLimitDepthThreshold: int
  }
  static member Default =
    {
      SingularityLowThreshold = 5
      RangeLowThreshold = 15
      ClusteringEnabled = true
      ClusteringTableSampleSize = 1000
      ClusteringMaxClusterWeight = 15.0
      ClusteringMergeThreshold = 0.1
      PrecisionLimitRowFraction = 10000
      PrecisionLimitDepthThreshold = 15
    }

type AnonymizationParams =
  {
    TableSettings: Map<string, TableSettings>
    Salt: byte[]
    Suppression: LowCountParams

    // Count params
    OutlierCount: Interval
    TopCount: Interval
    LayerNoiseSD: float
  }
  static member Default =
    {
      TableSettings = Map.empty
      Salt = [||]
      Suppression = SuppressionParams.Default
      OutlierCount = Interval.Default
      TopCount = Interval.Default
      LayerNoiseSD = 1.0
    }

type AnonymizationContext = { BucketSeed: Hash; AnonymizationParams: AnonymizationParams }

// ----------------------------------------------------------------
// Adaptive Buckets
// ----------------------------------------------------------------

let private parseBoolean (str: string) =
  match str.Trim().ToLower() with
  | "1"
  | "t"
  | "true" -> Boolean true
  | "0"
  | "f"
  | "false" -> Boolean false
  | _ -> Null

let private parseInteger (str: string) =
  match Int64.TryParse(str) with
  | true, value -> Integer value
  | _ -> Null

let private parseDouble (str: string) =
  match Double.TryParse(str, NumberStyles.Float ||| NumberStyles.AllowThousands, NumberFormatInfo.InvariantInfo) with
  | true, value -> Real value
  | _ -> Null

let private parseTimestamp (str: string) =
  match DateTime.TryParse(str, CultureInfo.InvariantCulture, DateTimeStyles.None) with
  | true, value -> Timestamp value
  | _ -> Null

let parseField columnType (value: string) =
  match columnType with
  | BooleanType -> parseBoolean value
  | IntegerType -> parseInteger value
  | RealType -> parseDouble value
  | TimestampType -> parseTimestamp value
  | StringType -> String value
  | _ -> failwith "Unsupported type!"

let TIMESTAMP_REFERENCE = DateTime(1800, 1, 1, 0, 0, 0, DateTimeKind.Utc)

open SynDiffix.Range

type MicrodataValue = ValueTuple<Value, float>
type MicrodataRow = MicrodataValue array

let inline microdataRowToRow (microdataRow: MicrodataRow) : Row = microdataRow |> Array.map (vfst)

type IDataConvertor =
  abstract ColumnType: ExpressionType
  // Casts a `Value` to a `float` in order to match it against a `Range`.
  abstract ToFloat: Value -> float
  // Generates a synthetic `Value` from an anonymized `Range`.
  abstract FromRange: Range -> MicrodataValue

type IDataConvertors = IDataConvertor array
