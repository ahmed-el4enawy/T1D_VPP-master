"""
build_midnight_cgm_stats_v5.py

Per-participant midnight CGM mean/SD from T1DEXI's LB.xpt, for the paper's
Section 2.2.1 initial-glucose step:

    "For each 7-day meal scenario, five different initial glucose values
     were drawn from a normal distribution, with the mean and standard
     deviation calculated using CGM values at midnight from the participant
     corresponding to each meal scenario. The mean of the midnight CGM was
     156 +/- 45 mg/dL."

CHANGES vs v4
-------------
1. MIN_DAYS_PER_SUBJECT LOWERED 5 -> 2. A sample SD is undefined at n=1
   (division by zero under pandas' default ddof=1) and first becomes
   defined at n=2 -- that is the only thing this number is derived from,
   not a claim that n=2 gives a trustworthy SD. See the inline comment at
   the constant for the coverage-vs-precision tradeoff this creates and
   why the SD-distribution report further down is what actually tells you
   whether it was a reasonable choice for this data. Still a
   reconstruction (A13), not a paper requirement, same as v4.

CHANGES vs v3
-------------
1. UNKNOWN UNITS ARE NOW A HARD ERROR. v3 fell through to
   "Proceeding UNFILTERED" with a warning. mmol/L read as mg/dL is an 18x
   error on the statistic that seeds every initial glucose in all 46,200
   scenarios, and a warning in a long log is not a safeguard. If the units
   cannot be positively verified as mg/dL the script now refuses to write
   a file. Set ALLOW_UNVERIFIED_UNITS = True only after looking at the
   printed unit values yourself.

2. BOTH n_days AND n_readings ARE WRITTEN. The existing
   load_midnight_cgm_stats.m read T.n_readings; v3 renamed the column to
   n_days, which would have thrown "Unrecognized table variable name" on
   the first line of the pilot. The CSV now carries both (identical
   values) so old and new loaders both work.

3. USUBJID NORMALISATION, identical to build_day_scenario_library_v4.py
   and load_midnight_cgm_stats.m. If pyreadstat returns USUBJID as
   numeric, one file can end up with "12.0" and the other with "12", in
   which case NO participant matches and every scenario silently falls
   back to the pooled 156 +/- 45.

UNCHANGED from v3 (all correct): CGM-only records, the date-only LBDTC
guard, one reading per participant per midnight event, pre-00:00 readings
assigned to the next midnight, pooled mean/SD reported as the statistic
comparable to 156 +/- 45, SD distribution reporting, no SD clamp.

RECONSTRUCTION NOTE: neither MIDNIGHT_WINDOW_MIN nor MIN_DAYS_PER_SUBJECT
is specified by the paper. The paper says only "CGM values at midnight".
Both are quality-control choices of ours and are recorded as assumption
A13 in the generator header.

SETUP:  pip install pyreadstat pandas numpy
USAGE:  python build_midnight_cgm_stats_v5.py LB.xpt midnight_cgm_stats.csv [day_scenario_library.csv]
"""

import re
import sys

import numpy as np
import pandas as pd
import pyreadstat

_INT_DOT = re.compile(r"^(-?\d+)\.0*$")

MIDNIGHT_WINDOW_MIN = 15          # RECONSTRUCTION (A13)
MIN_DAYS_PER_SUBJECT = 2          # RECONSTRUCTION (A13), lowered from 5.
# A sample SD is mathematically undefined at n=1 (pandas' default ddof=1
# divides by n-1=0 -> NaN; the existing `dropna(subset=["midnight_std"])`
# below already drops those regardless of this threshold) and first
# becomes DEFINED at n=2. That is the only thing 2 is derived from -- it
# is not a quality bar, just the floor below which "midnight SD" has no
# meaning at all. An SD computed from exactly 2 or 3 points is still a
# very noisy estimate of that participant's true midnight variability
# (their 5 initial-glucose variants will be as unstable as that estimate
# is), so lowering this from 5 trades participant-specific coverage
# (fewer participants forced into the pooled 156+/-45 fallback) against
# per-participant SD precision (more participants whose SD estimate rests
# on very few observations). The SD-distribution report below (p5/p25/p50/
# p75/p95, and the "SD < 15" / "SD > 80" counts) is what actually shows
# which of those two effects dominates for your data -- inspect it rather
# than assuming this number is safe. Neither 2 nor 5 is a paper
# requirement; the paper specifies only "CGM values at midnight."

SD_FLOOR = None                   # leave as None: no clamp (correct default)
SD_CEIL = None

ALLOW_UNVERIFIED_UNITS = False    # CHANGE 1: do not flip this casually

PAPER_MEAN, PAPER_SD = 156.0, 45.0

MGDL_TOKENS = {"mg/dl", "mg/dl.", "mgdl", "milligram per deciliter", "mg per dl"}
MMOL_TOKENS = {"mmol/l", "mmoll", "millimole per liter", "mmol per l"}


def normalise_id(v) -> str:
    """CHANGE 3. Must match normalise_id() in load_midnight_cgm_stats.m and
    the matching helper in the other build script. Only the "12.0" -> "12"
    float artefact is stripped; zero-padded ids like "0012" are preserved,
    because "0012" and "12" can be different subjects."""
    s = str(v).strip()
    if not s:
        return ""
    m = _INT_DOT.match(s)
    return m.group(1) if m else s


def extract_cgm_events(xpt_path: str, chunksize: int = 200_000) -> pd.DataFrame:
    cols = ["USUBJID", "LBCAT", "LBTESTCD", "LBSTRESN", "LBSTRESU", "LBORRESU", "LBDTC"]
    try:
        reader = pyreadstat.read_file_in_chunks(
            pyreadstat.read_xport, xpt_path, chunksize=chunksize, usecols=cols,
        )
    except Exception:
        cols = [c for c in cols if c != "LBORRESU"]
        reader = pyreadstat.read_file_in_chunks(
            pyreadstat.read_xport, xpt_path, chunksize=chunksize, usecols=cols,
        )

    pieces, total_rows, total_matches, chunk_num = [], 0, 0, 0
    cat_seen, test_seen = {}, {}

    for df, _meta in reader:
        chunk_num += 1
        total_rows += len(df)
        for c, n in df["LBCAT"].astype(str).value_counts().items():
            cat_seen[c] = cat_seen.get(c, 0) + int(n)
        for c, n in df["LBTESTCD"].astype(str).value_counts().items():
            test_seen[c] = test_seen.get(c, 0) + int(n)

        m = df["LBTESTCD"].astype(str).str.upper() == "GLUC"
        m &= df["LBCAT"].astype(str).str.upper().str.contains("CGM", na=False)

        matches = df[m].copy()
        if len(matches):
            pieces.append(matches)
            total_matches += len(matches)

        if chunk_num % 10 == 0:
            print(f"...scanned {total_rows:,} rows, {total_matches:,} CGM GLUC so far")

    print(f"\nDONE. rows={total_rows:,}  CGM glucose records={total_matches:,}")
    print("LBCAT values present (top 10):")
    for c, n in sorted(cat_seen.items(), key=lambda kv: -kv[1])[:10]:
        print(f"   {c:22s} {n:>12,}")
    print("LBTESTCD values present (top 10):")
    for c, n in sorted(test_seen.items(), key=lambda kv: -kv[1])[:10]:
        print(f"   {c:22s} {n:>12,}")

    if not pieces:
        raise RuntimeError(
            "No CGM GLUC records. Check the LBCAT / LBTESTCD values above."
        )
    return pd.concat(pieces, ignore_index=True)


def apply_unit_filter(df: pd.DataFrame) -> pd.DataFrame:
    """CHANGE 1: verify mg/dL or refuse to continue."""
    for col in ("LBSTRESU", "LBORRESU"):
        if col not in df.columns:
            continue
        u = df[col].astype(str).str.strip().str.lower()
        present = sorted(set(u[(u != "") & (u != "nan")]))
        if not present:
            print(f"Unit column {col} is empty; trying the next one.")
            continue
        print(f"Unit values in {col}: {present[:10]}")

        keep = u.isin(MGDL_TOKENS)
        n_mmol = int(u.isin(MMOL_TOKENS).sum())
        if n_mmol:
            print(f"  !! {n_mmol:,} rows are mmol/L. They are being EXCLUDED, "
                  "not converted. If most of your data is mmol/L, convert "
                  "(mg/dL = 18.018 x mmol/L) before using this script.")
        if keep.sum() == 0:
            print(f"  !! no mg/dL rows found in {col}.")
            continue
        print(f"  keeping {keep.sum():,} / {len(df):,} mg/dL rows")
        return df[keep]

    # CHANGE 1: the fallback that used to proceed unfiltered
    msg = (
        "Could not verify glucose units as mg/dL from LBSTRESU or LBORRESU.\n"
        "Refusing to build participant midnight statistics with unknown units: "
        "mmol/L read as mg/dL is an 18x error on the mean and SD that seed "
        "every initial glucose value in all 46,200 scenarios.\n"
        "Inspect the unit values printed above. If they really are mg/dL under "
        "a name this script does not recognise, add it to MGDL_TOKENS. Only as "
        "a last resort set ALLOW_UNVERIFIED_UNITS = True."
    )
    if not ALLOW_UNVERIFIED_UNITS:
        raise RuntimeError(msg)
    print("!! ALLOW_UNVERIFIED_UNITS is True. Proceeding unfiltered. "
          "Check the pooled mean below: ~156 means mg/dL, ~8.7 means mmol/L.")
    return df


def build_midnight_stats(cgm: pd.DataFrame):
    df = apply_unit_filter(cgm.copy())

    raw = df["LBDTC"].astype(str).str.strip()
    has_time = raw.str.len() > 10
    n_dateonly = int((~has_time).sum())
    if n_dateonly:
        print(f"Dropping {n_dateonly:,} date-only LBDTC records "
              "(they would fake exact-midnight readings and win every "
              "closest-to-midnight sort).")
        df = df[has_time]

    df = df.copy()
    df["USUBJID"] = df["USUBJID"].map(normalise_id)          # CHANGE 3
    df["dt"] = pd.to_datetime(df["LBDTC"], errors="coerce")
    df = df.dropna(subset=["dt"])
    df = df[(df["LBSTRESN"] > 20) & (df["LBSTRESN"] < 600)]

    mins = df["dt"].dt.hour * 60 + df["dt"].dt.minute
    df["dist"] = mins.apply(lambda m: min(m, 1440 - m))

    next_day = mins > 720
    df["mid_date"] = df["dt"].dt.normalize()
    df.loc[next_day, "mid_date"] = df.loc[next_day, "mid_date"] + pd.Timedelta(days=1)

    df = df[df["dist"] <= MIDNIGHT_WINDOW_MIN]
    df = df.sort_values("dist")
    daily = df.groupby(["USUBJID", "mid_date"], as_index=False).first()
    print(f"\nSelected {len(daily):,} participant-days (1 reading each, "
          f"within +/-{MIDNIGHT_WINDOW_MIN} min of midnight).")

    stats = (daily.groupby("USUBJID")["LBSTRESN"]
                  .agg(["mean", "std", "count"]).reset_index())
    stats.columns = ["usubjid", "midnight_mean", "midnight_std", "n_days"]

    before = len(stats)
    stats = stats[stats["n_days"] >= MIN_DAYS_PER_SUBJECT]
    stats = stats.dropna(subset=["midnight_std"])
    stats = stats[stats["midnight_std"] > 0]
    print(f"Kept {len(stats):,} / {before:,} subjects with >= "
          f"{MIN_DAYS_PER_SUBJECT} midnight days (RECONSTRUCTION, not a "
          "paper requirement).")

    sd = stats["midnight_std"]
    print("\nPer-subject midnight SD (this is what separates the five "
          "initial-glucose variants):")
    for q in (5, 25, 50, 75, 95):
        print(f"   p{q:<3d} = {np.percentile(sd, q):6.1f} mg/dL")
    print(f"   subjects with SD < 15 mg/dL : {(sd < 15).sum():,} "
          "(their 5 variants will be near-identical)")
    print(f"   subjects with SD > 80 mg/dL : {(sd > 80).sum():,}")
    print(f"Days per subject: min {stats['n_days'].min()}, "
          f"median {stats['n_days'].median():.0f}, max {stats['n_days'].max()}")

    if SD_FLOOR is not None or SD_CEIL is not None:
        lo = SD_FLOOR if SD_FLOOR is not None else 0.0
        hi = SD_CEIL if SD_CEIL is not None else np.inf
        n_clamped = int(((sd < lo) | (sd > hi)).sum())
        stats["midnight_std"] = sd.clip(lower=lo, upper=hi)
        print(f"\nRECONSTRUCTION: clamped {n_clamped:,} subject SDs to "
              f"[{lo}, {hi}] mg/dL. The paper specifies no such clamp.")

    # CHANGE 2: keep both names so either loader works
    stats["n_readings"] = stats["n_days"]

    return stats, daily["LBSTRESN"]


def cross_check_meal_library(stats: pd.DataFrame, lib_path: str) -> None:
    lib = pd.read_csv(lib_path, usecols=["usubjid"])
    meal_subj = {normalise_id(x) for x in lib["usubjid"]}
    cgm_subj = {normalise_id(x) for x in stats["usubjid"]}
    both = meal_subj & cgm_subj
    print("\n" + "=" * 70)
    print("PARTICIPANT CROSS-CHECK against the meal library")
    print("=" * 70)
    print(f"  meal-library participants : {len(meal_subj):,}")
    print(f"  midnight-CGM participants : {len(cgm_subj):,}")
    print(f"  in BOTH                   : {len(both):,} "
          f"({100*len(both)/max(len(meal_subj),1):.1f}% of meal participants)")
    if len(both) == 0:
        print("  *** ZERO overlap. Almost always an id-format mismatch (e.g. "
              '"12" vs "12.0"). Both v4 scripts normalise ids identically, so '
              "if this is still zero the two files are from different studies "
              "or different phases. ***")
    missing = len(meal_subj) - len(both)
    if missing:
        print(f"  !! {missing:,} meal participants have NO midnight CGM stats.")
        print("     Those scenarios fall back to the pooled 156 +/- 45, which")
        print("     drops the participant-specific design the paper describes.")
        print("     The generator reports the realised fallback fraction as")
        print("     meta.used_pooled -- aim for 0 before production.")
    if len(both) < 0.8 * len(meal_subj):
        print("     Under 80% coverage usually means FAMLPM.xpt and LB.xpt come")
        print("     from DIFFERENT T1DEXI phases. The paper uses the PILOT phase")
        print("     for meal scenarios.")


import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build midnight CGM stats from T1DEXI LB.xpt")
    parser.add_argument("xpt_in", type=str, help="Input LB.xpt file")
    parser.add_argument("csv_out", type=str, nargs='?', default="midnight_cgm_stats.csv", help="Output CSV file")
    parser.add_argument("lib_in", type=str, nargs='?', default=None, help="Optional day_scenario_library.csv for cross-check")
    parser.add_argument("--manifest", type=str, default="midnight_cgm_stats_manifest.json", help="Output JSON manifest")
    args = parser.parse_args()

    xpt_in = args.xpt_in
    csv_out = args.csv_out
    lib_in = args.lib_in

    cgm = extract_cgm_events(xpt_in)
    stats, pooled = build_midnight_stats(cgm)

    pm, ps = pooled.mean(), pooled.std(ddof=1)
    print("\nPOOLED midnight CGM (the statistic comparable to the paper's "
          f"{PAPER_MEAN:.0f} +/- {PAPER_SD:.0f} mg/dL):")
    print(f"  mean = {pm:.1f}   (paper {PAPER_MEAN:.0f}, delta {pm-PAPER_MEAN:+.1f})")
    print(f"  sd   = {ps:.1f}   (paper {PAPER_SD:.0f}, delta {ps-PAPER_SD:+.1f})")
    print(f"  n    = {len(pooled):,}")
    if 5 < pm < 15:
        raise RuntimeError(
            f"Pooled mean is {pm:.1f}, which is mmol/L, not mg/dL. Refusing to "
            "write the file. Convert with mg/dL = 18.018 x mmol/L."
        )
    if abs(pm - PAPER_MEAN) > 20 or abs(ps - PAPER_SD) > 20:
        print("  !! more than 20 mg/dL off the paper. Check the study phase, "
              "the units, and the date-only guard above before using this file.")

    stats.to_csv(csv_out, index=False)
    print(f"\nSaved {len(stats):,} per-subject midnight CGM stats to {csv_out}")
    print("  columns: usubjid, midnight_mean, midnight_std, n_days, n_readings")
    print("  (n_readings duplicates n_days so both the old and new MATLAB "
          "loaders work)")

    print("\n(For reference only - NOT comparable to the paper:)")
    print(f"  mean of subject means = {stats['midnight_mean'].mean():.1f}")
    print(f"  mean of subject SDs   = {stats['midnight_std'].mean():.1f}")

    if lib_in:
        cross_check_meal_library(stats, lib_in)
        
    # Save manifest
    manifest = {
        "file": csv_out,
        "rows": len(stats),
        "size_bytes": os.path.getsize(csv_out),
        "pooled_mean": pm,
        "pooled_std": ps
    }
    with open(args.manifest, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {args.manifest}")
