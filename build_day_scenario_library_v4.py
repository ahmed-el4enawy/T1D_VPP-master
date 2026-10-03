"""
build_day_scenario_library_v5.py

Builds the daily meal-scenario library the MATLAB generator samples from,
out of T1DEXI's FAMLPM.xpt (FATESTCD == 'DCARBT').

CHANGES vs v4
-------------
4. ROUNDING NO LONGER ZEROES OUT SMALL LEGITIMATE CARB VALUES.
   v4 already drops rows with FASTRESN <= 0 (clean_events), which correctly
   removes the ~17,700 truly-zero-carb meal entries found in FAMLPM.xpt
   (real meal events -- Breakfast/Lunch/Dinner/Snack -- logged with 0 g,
   confirmed via FAOBJ). But ~51 SEPARATE rows have a small POSITIVE carb
   value (e.g. 0.04, 0.07 g) that PASSES that filter, then gets silently
   rounded to "0.0" by round(c, 1) when writing carb_grams. That
   reintroduces the exact same downstream failure
   (load_day_scenario_library.m: "Row N has a non-positive carb value")
   on a different row, for a different reason (rounding, not a genuine
   zero). Formatting with 6 significant figures instead of 1 decimal place
   preserves these small values (0.04 stays "0.04") without changing any
   normal-sized value's precision in practice.

CHANGES vs v3 (carried over, unchanged)
----------------------------------------
1. CLUSTER_MIN DEFAULT IS NOW 0 (no clustering).
   v3 defaulted to 15-minute clustering and told you to pick whichever
   setting reproduced the paper's 1-9 meals/day. That criterion is
   circular: it tunes preprocessing against a published percentile rather
   than against the structure of the source data, so you could "reproduce"
   the number while misrepresenting how the authors counted meals.
   The T1DEXI app collected self-reported meal/snack entries with an
   estimated carbohydrate amount per entry, and independent work on the
   raw files treats FAMLPM DCARBT rows as timestamped intake events. So
   meal-level is the default reading until the raw file proves otherwise.

2. STRUCTURAL FOOD-ITEM TEST (replaces the percentile criterion).
   Whether a DCARBT row is a meal or a food item is a question about the
   data, and the data can answer it:
       - fraction of records sharing an EXACT timestamp with another
         record for the same participant  (food-item rows cluster at one
         logged time; meal rows do not)
       - distribution of gaps between consecutive records within a day
         (meal-level -> hour-scale gaps; item-level -> a spike near 0)
       - FAOBJ cardinality per (participant, timestamp)
   Enable clustering ONLY if these say item-level.

3. USUBJID NORMALISATION. If pyreadstat returns USUBJID as numeric, pandas
   can write "12.0" here and "12" in the CGM statistics file (or vice
   versa), in which case NO participant ever matches and every scenario
   silently falls back to the pooled 156 +/- 45. Both build scripts and
   the MATLAB loader now normalise identically.

UNCHANGED from v3 (all correct): duplicate removal, the date-only FADTC
guard, deterministic sort, day_id, the 1-9 / 30-359 daily filters, and the
removal of v1's undocumented per-meal <300 g cap.

SETUP:  pip install pyreadstat pandas numpy
USAGE:  python build_day_scenario_library_v5.py FAMLPM.xpt day_scenario_library.csv
"""

import re
import sys

import numpy as np
import pandas as pd
import pyreadstat

_INT_DOT = re.compile(r"^(-?\d+)\.0*$")

TARGET_TESTCD = "DCARBT"

# Paper's documented 5th-95th percentile limits (source-backed)
MIN_MEALS, MAX_MEALS = 1, 9
MIN_CARBS, MAX_CARBS = 30, 359

# CHANGE 1: 0 = no clustering. Only raise this if the structural test in
# food_item_evidence() shows the rows are food-item level.
CLUSTER_MIN = 0


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


def extract_carb_events(xpt_path: str, chunksize: int = 200_000) -> pd.DataFrame:
    cols = ["USUBJID", "FATESTCD", "FAOBJ", "FASTRESN", "FADTC"]

    reader = pyreadstat.read_file_in_chunks(
        pyreadstat.read_xport, xpt_path, chunksize=chunksize, usecols=cols,
    )

    pieces, total_rows, total_matches, chunk_num = [], 0, 0, 0
    testcd_seen = {}

    for df, _meta in reader:
        chunk_num += 1
        total_rows += len(df)
        for code, cnt in df["FATESTCD"].astype(str).value_counts().items():
            testcd_seen[code] = testcd_seen.get(code, 0) + int(cnt)

        matches = df[df["FATESTCD"] == TARGET_TESTCD].copy()
        if len(matches):
            pieces.append(matches)
            total_matches += len(matches)

        if chunk_num % 10 == 0:
            print(f"...scanned {total_rows:,} rows, {total_matches:,} DCARBT so far")

    print(f"\nDONE scanning. rows={total_rows:,}  DCARBT={total_matches:,}")
    print("FATESTCD values present (top 15):")
    for code, cnt in sorted(testcd_seen.items(), key=lambda kv: -kv[1])[:15]:
        star = "  <- selected" if code == TARGET_TESTCD else ""
        print(f"   {code:14s} {cnt:>10,}{star}")

    if not pieces:
        raise RuntimeError(
            f"No {TARGET_TESTCD} records. Check the list above for the right code."
        )
    return pd.concat(pieces, ignore_index=True)


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)

    raw = df["FADTC"].astype(str).str.strip()
    has_time = raw.str.len() > 10
    n_dateonly = int((~has_time).sum())
    if n_dateonly:
        print(f"Dropping {n_dateonly:,} date-only FADTC records (no time of day).")
        df = df[has_time]

    df = df.copy()
    df["USUBJID"] = df["USUBJID"].map(normalise_id)          # CHANGE 3
    df["dt"] = pd.to_datetime(df["FADTC"], errors="coerce")
    df = df.dropna(subset=["dt"])

    # Drop non-positive carb values. Confirmed by inspection: these are
    # real logged meal events (FAOBJ = Breakfast/Lunch/Dinner/Snack) with
    # FASTRESN == 0.0 in the raw file itself, not a parsing artefact -- a
    # 0 g "meal" has no effect on the glucose ODE and is not a usable
    # scenario input, so it is dropped here rather than reaching the CSV.
    n_before_carb_filter = len(df)
    df = df[df["FASTRESN"] > 0]
    n_zero_or_neg = n_before_carb_filter - len(df)
    if n_zero_or_neg:
        print(f"Dropping {n_zero_or_neg:,} records with carb value <= 0 "
              "(real logged meals recorded with 0 g carbs).")

    print(f"Usable carb records (pre-dedup): {len(df):,} (from {n0:,})")
    print(f"Date range: {df['dt'].min()} to {df['dt'].max()}")
    print(f"Participants with carb data: {df['USUBJID'].nunique():,}")
    return df


def dedupe_events(df: pd.DataFrame) -> pd.DataFrame:
    """v5: dedup key includes FAOBJ, and this runs AFTER
    food_item_evidence() rather than before it.

    v4 called drop_duplicates(subset=[USUBJID, dt, FASTRESN]) inside
    clean_events(), then fed the ALREADY-deduplicated frame into
    food_item_evidence(). But two legitimate, distinct food-item records
    can share participant + exact timestamp + carb amount while differing
    only in FAOBJ (e.g. two different 20 g items logged in the same
    app-batch entry) -- exactly the signal (a) and (c) in
    food_item_evidence() are designed to detect. Deduplicating on
    [USUBJID, dt, FASTRESN] before running that test silently erases the
    evidence the test needs, biasing CLUSTER_MIN's decision toward
    "meal-level" regardless of the true structure. Now: run the structural
    test on the cleaned-but-not-deduplicated events, THEN dedupe (with
    FAOBJ in the key, so it only removes true exact duplicates, not
    distinct items that happen to share a timestamp and amount)."""
    before = len(df)
    dedupe_cols = ["USUBJID", "dt", "FASTRESN"]
    if "FAOBJ" in df.columns:
        dedupe_cols.append("FAOBJ")
    df = df.drop_duplicates(subset=dedupe_cols)
    if before != len(df):
        print(f"Dropped {before - len(df):,} exact duplicate carb records "
              f"(key: {dedupe_cols}).")
    return df


# ---------------------------------------------------------------------------
# CHANGE 2: structural evidence, not percentile matching
# ---------------------------------------------------------------------------
def food_item_evidence(df: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("IS ONE DCARBT ROW A MEAL OR A FOOD ITEM?")
    print("=" * 74)
    print("This is a question about the data, so answer it from the data --")
    print("NOT by picking whichever clustering reproduces the paper's 1-9.\n")

    # (a) exact-timestamp collisions
    grp = df.groupby(["USUBJID", "dt"]).size()
    n_colliding = int(grp[grp > 1].sum())
    frac = 100.0 * n_colliding / max(len(df), 1)
    print(f"  (a) records sharing an EXACT timestamp with another record")
    print(f"      for the same participant : {n_colliding:,} / {len(df):,} ({frac:.1f}%)")
    print(f"      max records at one timestamp : {int(grp.max())}")

    # (b) within-day gaps between consecutive records
    d = df.sort_values(["USUBJID", "dt"]).copy()
    d["date"] = d["dt"].dt.date
    gaps = (d.groupby(["USUBJID", "date"])["dt"]
              .diff().dt.total_seconds().div(60.0).dropna())
    if len(gaps):
        print(f"\n  (b) gap between consecutive records within a day (minutes)")
        for q in (5, 25, 50, 75, 95):
            print(f"      p{q:<3d} = {np.percentile(gaps, q):8.1f}")
        print(f"      fraction of gaps under 15 min : "
              f"{100.0*np.mean(gaps < 15):.1f}%")
        print(f"      fraction of gaps under  2 min : "
              f"{100.0*np.mean(gaps < 2):.1f}%")

    # (c) FAOBJ cardinality at a single timestamp
    if "FAOBJ" in df.columns:
        obj_per_stamp = df.groupby(["USUBJID", "dt"])["FAOBJ"].nunique()
        print(f"\n  (c) distinct FAOBJ values at one participant-timestamp")
        print(f"      median {obj_per_stamp.median():.0f}, "
              f"max {obj_per_stamp.max():.0f}")
        vals = df["FAOBJ"].astype(str).value_counts()
        print(f"      distinct FAOBJ values overall : {len(vals):,}")
        print(f"      most common: {list(vals.index[:5])}")

    print("\n  READING IT:")
    print("   MEAL-LEVEL   -> (a) near 0%, (b) gaps concentrated at hour scale,")
    print("                   (c) one FAOBJ per timestamp.  Keep CLUSTER_MIN = 0.")
    print("   ITEM-LEVEL   -> (a) large, (b) a heavy spike under 2 min,")
    print("                   (c) several FAOBJ per timestamp.  Only then raise")
    print("                   CLUSTER_MIN, and record it as a reconstruction.")
    print("=" * 74)


def cluster_meals(g: pd.DataFrame, cluster_min: int) -> pd.DataFrame:
    if cluster_min <= 0:
        out = g[["t_of_day_min", "FASTRESN"]].rename(
            columns={"FASTRESN": "carbs", "t_of_day_min": "t"}
        )
        return out.sort_values("t")
    g = g.sort_values("t_of_day_min")
    times = g["t_of_day_min"].to_numpy()
    carbs = g["FASTRESN"].to_numpy()
    out_t, out_c = [], []
    for t, c in zip(times, carbs):
        if out_t and (t - out_t[-1]) <= cluster_min:
            out_c[-1] += c
        else:
            out_t.append(t)
            out_c.append(c)
    return pd.DataFrame({"t": out_t, "carbs": out_c})


def build_rows(df: pd.DataFrame, cluster_min: int) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["dt"].dt.date
    df["day_key"] = df["USUBJID"].astype(str) + "_" + df["date"].astype(str)
    df["t_of_day_min"] = df["dt"].dt.hour * 60 + df["dt"].dt.minute

    rows = []
    for day_key, g in df.groupby("day_key"):
        meals = cluster_meals(g, cluster_min)
        rows.append({
            "day_key": day_key,
            "usubjid": g["USUBJID"].iloc[0],
            "num_meals": len(meals),
            "total_carbs": float(meals["carbs"].sum()),
            # CHANGE 4: 6 significant figures instead of round(c, 1). A
            # small positive carb value (e.g. 0.04 g) survives the > 0
            # filter in clean_events() but round(0.04, 1) -> 0.0, which
            # would write "0.0" here and reintroduce the exact
            # "non-positive carb value" failure in
            # load_day_scenario_library.m on a different row, for a
            # rounding reason rather than a genuine zero. f"{x:.6g}"
            # preserves 0.04 as "0.04" while leaving normal-sized values
            # (tens to hundreds of grams) effectively unchanged in
            # precision for this use case.
            "carb_grams": ";".join(
                f"{float(c):.6g}" for c in meals["carbs"]
            ),
            "time_min": ";".join(str(int(t)) for t in meals["t"]),
            "_sortkey": (str(g["USUBJID"].iloc[0]), str(g["date"].iloc[0])),
        })
    return pd.DataFrame(rows)


def percentile_report(raw: pd.DataFrame, label: str) -> None:
    print(f"  [{label}] meals/day 5th-95th : "
          f"{raw['num_meals'].quantile(0.05):.0f} - "
          f"{raw['num_meals'].quantile(0.95):.0f}  (paper: {MIN_MEALS}-{MAX_MEALS})")
    print(f"  [{label}] g/day     5th-95th : "
          f"{raw['total_carbs'].quantile(0.05):.0f} - "
          f"{raw['total_carbs'].quantile(0.95):.0f}  (paper: {MIN_CARBS}-{MAX_CARBS})")
    print(f"  [{label}] mean carbs/day     : {raw['total_carbs'].mean():.0f} g")


def main():
    xpt_in = sys.argv[1]
    csv_out = sys.argv[2] if len(sys.argv) > 2 else "day_scenario_library.csv"

    events = clean_events(extract_carb_events(xpt_in))

    food_item_evidence(events)          # v5: BEFORE dedup, on raw structure

    events = dedupe_events(events)      # v5: dedup happens after the test

    print("\nPercentiles under both settings, FOR INFORMATION ONLY -- do not")
    print("choose CLUSTER_MIN from these. Choose it from the evidence above.\n")
    raw_nc = build_rows(events, 0)
    percentile_report(raw_nc, "no clustering")
    raw_c15 = build_rows(events, 15)
    percentile_report(raw_c15, "15-min clusters")

    print(f"\nUsing CLUSTER_MIN = {CLUSTER_MIN} minutes "
          f"({'no clustering (default)' if CLUSTER_MIN == 0 else 'RECONSTRUCTION'}).")
    raw = raw_nc if CLUSTER_MIN == 0 else build_rows(events, CLUSTER_MIN)

    library = (raw.sort_values("_sortkey")
                  .drop(columns=["_sortkey"])
                  .reset_index(drop=True))

    before = len(library)
    library = library[
        (library["num_meals"] >= MIN_MEALS) & (library["num_meals"] <= MAX_MEALS) &
        (library["total_carbs"] >= MIN_CARBS) & (library["total_carbs"] <= MAX_CARBS)
    ].reset_index(drop=True)
    library.insert(0, "day_id", range(1, len(library) + 1))

    print(f"\nKept {len(library):,} / {before:,} daily scenarios after the paper's "
          f"filters ({MIN_MEALS}-{MAX_MEALS} meals/day, {MIN_CARBS}-{MAX_CARBS} g/day).")
    if len(library) == 0:
        raise RuntimeError(
            "The filters removed every day. Look at the percentiles above: if "
            "meals/day sits far above 9, the rows are probably food-item level "
            "(see the structural test), or this is the wrong study phase."
        )

    per_subj = library.groupby("usubjid").size()
    print(f"Participants in library  : {len(per_subj):,}")
    print(f"Days per participant     : min {per_subj.min()}, "
          f"median {per_subj.median():.0f}, max {per_subj.max()}")
    print(f"Participants with 1 day  : {(per_subj == 1).sum():,} "
          "(each produces a 7-day scenario repeating one day 7x)")
    print(f"Mean carbs/day in library: {library['total_carbs'].mean():.0f} g")

    library.to_csv(csv_out, index=False)
    print(f"\nSaved to {csv_out}")
    print("NOTE: meta.day_idx in the generated .mat refers to ROW POSITIONS in")
    print("this file, and day_id must stay equal to the row number. The MATLAB")
    print("loader asserts this. Keep the file under version control; do not")
    print("re-sort it by hand.")


if __name__ == "__main__":
    main()
