module SynDiffix.Anonymizer

open System
open System.Security.Cryptography

type ContributionsState = { AidContributions: Dictionary<AidHash, float>; mutable UnaccountedFor: float }

type CountResult = { AnonymizedCount: int64; NoiseSD: float }

// ----------------------------------------------------------------
// Random & noise
// ----------------------------------------------------------------

// The noise seeds are hash values.
// From each seed we generate a single random value, with either a uniform or a normal distribution.
// Any decent hash function should produce values that are uniformly distributed over the output space.
// Hence, we only need to limit the seed to the requested interval to get a uniform random integer.
// To get a normal random float, we use the Box-Muller method on two uniformly distributed integers.

let private randomUniform (interval: Interval) (seed: Hash) =
  let randomUniform = uint32 ((seed >>> 32) ^^^ seed)
  // While using modulo to bound values produces biased output, we are using very small ranges
  // (typically less than 10), for which the bias is insignificant.
  let boundedRandomUniform = randomUniform % uint32 (interval.Upper - interval.Lower + 1)
  interval.Lower + int boundedRandomUniform

let private randomNormal stdDev (seed: Hash) =
  let u1 = float (uint32 seed) / float UInt32.MaxValue
  let u2 = float (uint32 (seed >>> 32)) / float UInt32.MaxValue
  let randomNormal = Math.Sqrt(-2.0 * log u1) * Math.Sin(2.0 * Math.PI * u2)
  stdDev * randomNormal

let private cryptoHashSaltedSeed salt (seed: Hash) : Hash =
  use sha256 = SHA256.Create()
  let seedBytes = BitConverter.GetBytes(seed)
  let hash = sha256.ComputeHash(Array.append salt seedBytes)
  BitConverter.ToUInt64(hash, 0)

let seedFromAidSet (aidSet: AidHash seq) = Seq.fold (^^^) 0UL aidSet

let private mixSeed (text: string) (seed: Hash) = text |> Hash.string |> ((^^^) seed)

let private generateNoise salt stepName stdDev noiseLayers =
  noiseLayers
  |> Seq.map (cryptoHashSaltedSeed salt >> mixSeed stepName >> randomNormal stdDev)
  |> Seq.reduce (+)

// ----------------------------------------------------------------
// Depth limiting
// ----------------------------------------------------------------

let noisyRowLimit salt (seed: Hash) nRows rowFraction =
  let rowLimitWithoutNoise = nRows / rowFraction

  // Select an integer between plus and minus 5% of `rowLimitWithoutNoise`.
  let noiseOffsetScale = rowLimitWithoutNoise / 20

  seed
  |> cryptoHashSaltedSeed salt
  |> mixSeed "precision_limit"
  |> randomUniform { Lower = -noiseOffsetScale; Upper = noiseOffsetScale }
  |> (+) rowLimitWithoutNoise

// ----------------------------------------------------------------
// AID processing
// ----------------------------------------------------------------

// Compacts flattening intervals to fit into the total count of contributors.
// Both intervals are reduced proportionally, with `topCount` taking priority.
// `None` is returned in case there's not enough AIDVs for a sensible flattening.
// `public` just to test the low-level algorithm
let private compactFlatteningIntervals outlierCount topCount totalCount =
  if totalCount < outlierCount.Lower + topCount.Lower then
    None
  else
    let totalAdjustment = outlierCount.Upper + topCount.Upper - totalCount

    let compactIntervals =
      if totalAdjustment <= 0 then
        outlierCount, topCount // no adjustment needed
      else
        // NOTE: at this point we know `0 < totalAdjustment <= outlierRange + topRange` (*)
        //       because `totalAdjustment = outlierCount.Upper + topCount.Upper - totalCount
        //                               <= outlierCount.Upper + topCount.Upper - outlierCount.Lower - topCount.Lower`
        let outlierRange = outlierCount.Upper - outlierCount.Lower
        let topRange = topCount.Upper - topCount.Lower
        // `topAdjustment` will be half of `totalAdjustment` rounded up, so it takes priority as it should
        let outlierAdjustment = totalAdjustment / 2
        let topAdjustment = totalAdjustment - outlierAdjustment

        // adjust, depending on how the adjustments "fit" in the ranges
        match outlierRange >= outlierAdjustment, topRange >= topAdjustment with
        | true, true ->
          // both ranges are compacted at same rate
          { outlierCount with Upper = outlierCount.Upper - outlierAdjustment },
          { topCount with Upper = topCount.Upper - topAdjustment }
        | false, true ->
          // `outlierCount` is compacted as much as possible by `outlierRange`, `topCount` takes the surplus adjustment
          { outlierCount with Upper = outlierCount.Lower },
          { topCount with Upper = topCount.Upper - totalAdjustment + outlierRange }
        | true, false ->
          // vice versa
          { outlierCount with Upper = outlierCount.Upper - totalAdjustment + topRange },
          { topCount with Upper = topCount.Lower }
        | false, false ->
          // Not possible. Otherwise `outlierRange + topRange < outlierAdjustment + topAdjustment = totalAdjustment` but we
          // knew the opposite was true in (*) above
          failwith "Impossible interval compacting."

    Some compactIntervals

type private AidCount = { FlattenedSum: float; Flattening: float; NoiseSD: float; Noise: float }

let inline private aidFlattening
  (anonContext: AnonymizationContext)
  (unaccountedFor: float)
  (aidContributions: (AidHash * ^Contribution) array)
  : AidCount option =
  let totalCount = aidContributions.Length
  let anonParams = anonContext.AnonymizationParams

  match compactFlatteningIntervals anonParams.OutlierCount anonParams.TopCount totalCount with
  | None -> None // not enough AIDVs for a sensible flattening
  | Some(outlierInterval, topInterval) ->
    let sortedAidContributions =
      aidContributions
      |> Array.sortByDescending (fun (aid, contribution) -> contribution, aid)

    let flatSeed =
      sortedAidContributions
      |> Seq.take (outlierInterval.Upper + topInterval.Upper)
      |> Seq.map fst
      |> seedFromAidSet
      |> cryptoHashSaltedSeed anonParams.Salt

    let outlierCount = flatSeed |> mixSeed "outlier" |> randomUniform outlierInterval
    let topCount = flatSeed |> mixSeed "top" |> randomUniform topInterval

    let topGroupSum =
      sortedAidContributions
      |> Seq.skip outlierCount
      |> Seq.take topCount
      |> Seq.sumBy snd

    let topGroupAverage = (float topGroupSum) / (float topCount)

    let flattening =
      sortedAidContributions
      |> Seq.take outlierCount
      |> Seq.map snd
      |> Seq.map (fun contribution -> (float contribution) - topGroupAverage)
      |> Seq.filter (fun contribution -> contribution > 0)
      |> Seq.sum

    let realSum = aidContributions |> Array.sumBy snd
    let flattenedUnaccountedFor = unaccountedFor - flattening |> max 0.
    let flattenedSum = float realSum - flattening
    let flattenedAvg = flattenedSum / float totalCount

    let noiseScale = max flattenedAvg (0.5 * topGroupAverage)
    let noiseSD = anonParams.LayerNoiseSD * noiseScale

    let noise =
      [ anonContext.BucketSeed; aidContributions |> Seq.map fst |> seedFromAidSet ]
      |> generateNoise anonParams.Salt "noise" noiseSD

    Some
      {
        FlattenedSum = flattenedSum + flattenedUnaccountedFor
        Flattening = flattening
        NoiseSD = noiseSD
        Noise = noise
      }

let private arrayFromDict (d: Dictionary<'a, 'b>) =
  d |> Seq.map (fun pair -> pair.Key, pair.Value) |> Seq.toArray

let private mapAidFlattening (anonContext: AnonymizationContext) perAidContributions =
  perAidContributions
  |> Array.map (fun aidState ->
    aidState.AidContributions
    |> arrayFromDict
    |> aidFlattening anonContext aidState.UnaccountedFor
  )

// Assumes that `byAidSum` is non-empty, meaning that there is at least one AID instance involved
let private anonymizedSum (byAidSum: AidCount seq) =
  let flattening =
    byAidSum
    // We might end up with multiple different flattened sums that have the same amount of flattening.
    // This could be the result of some AID values being null for one of the AIDs, while there were still
    // overall enough AIDs to produce a flattened sum.
    // In these case we want to use the largest flattened sum to minimize unnecessary flattening.
    |> Seq.maxBy (fun aggregate -> aggregate.Flattening, aggregate.FlattenedSum)

  let noise =
    byAidSum
    // For determinism, resolve draws using maximum absolute noise value.
    |> Seq.maxBy (fun aggregate -> aggregate.NoiseSD, Math.Abs(aggregate.Noise))

  (flattening.FlattenedSum + noise.Noise, noise.NoiseSD)

let private moneyRoundNoise noiseSD =
  if noiseSD = 0.0 then
    0.0
  else
    let roundingResolution = moneyRound (0.05 * noiseSD)

    (noiseSD / roundingResolution) |> ceil |> (*) roundingResolution

// ----------------------------------------------------------------
// Public API
// ----------------------------------------------------------------

/// Returns whether any of the AID value sets has a low count.
let isLowCount salt (lowCountParams: LowCountParams) (aidTrackers: (int64 * Hash) seq) =
  aidTrackers
  |> Seq.map (fun (count, seed) ->
    if count < lowCountParams.LowThreshold then
      true
    else
      let thresholdNoise = generateNoise salt "suppress" lowCountParams.LayerSD [ seed ]

      // `LowMeanGap` is the number of standard deviations between `LowThreshold` and desired mean
      let thresholdMean =
        lowCountParams.LowMeanGap * lowCountParams.LayerSD
        + float lowCountParams.LowThreshold

      let threshold = thresholdNoise + thresholdMean

      float count < threshold
  )
  |> Seq.reduce (||)

let countMultipleContributions (anonContext: AnonymizationContext) (contributions: ContributionsState array) =
  let byAid = mapAidFlattening anonContext contributions

  // If any of the AIDs had insufficient data to produce a sensible flattening
  // we have to abort anonymization.
  if byAid |> Array.exists ((=) None) then
    None
  else
    let (value, noiseSD) = byAid |> Array.choose id |> anonymizedSum

    Some
      {
        AnonymizedCount = value |> Math.roundAwayFromZero |> int64
        NoiseSD = moneyRoundNoise noiseSD
      }

let countSingleContributions (anonContext: AnonymizationContext) (count: int64) (seed: Hash) =
  [ anonContext.BucketSeed; seed ]
  |> generateNoise anonContext.AnonymizationParams.Salt "noise" anonContext.AnonymizationParams.LayerNoiseSD
  |> (+) (float count)
  |> Math.roundAwayFromZero
  |> int64
