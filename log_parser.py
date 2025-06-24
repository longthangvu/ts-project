#!/usr/bin/env python3
"""
extract_results.py

Walk logs/{model}/run.log, for each "start training" iteration extract:
  - model, data, seq_len, label_len, pred_len, time_budget, train_budget, itr
Then when the metrics dict appears, grab mae/mse/rmse/mape/mspe plus timers,
and write one CSV row per iteration.
"""

import os, re, csv, argparse

# Regex to pull args from the "start training" banner
START_RE = re.compile(
    r">+testing :\s*"
    # skip first 5 underscore-delimited chunks (forecastpfn, Exchange, 36, 18, 6)
    r"(?:[^_]+_){7}"
    # now capture the real model and data
    r"(?P<model>[^_]+)_"
    r"(?P<data>[^_]+)"
    # skip any number of extra _tokens_ before the sl<seq> piece
    r"(?:_[^_]+)*"
    r"_sl(?P<seq_len>\d+)_"
    r"ll(?P<label_len>\d+)_"
    r"pl(?P<pred_len>\d+)"
    # skip any extra tokens (e.g. dm512_nh8_…)
    r"(?:_[^_]+)*"
    # finally the test_<itr> instead of _itr_
    r"_test_(?P<itr>\d+)"
)

START_RE2 = re.compile(
    r">+testing :\s*"
    # skip first 5 underscore-delimited chunks (forecastpfn, Exchange, 36, 18, 6)
    r"(?:[^_]+_){5}"
    # now capture the real model and data
    r"(?P<model>[^_]+)_"
    r"(?P<data>[^_]+)"
    # skip any number of extra _tokens_ before the sl<seq> piece
    r"(?:_[^_]+)*"
    r"_sl(?P<seq_len>\d+)_"
    r"ll(?P<label_len>\d+)_"
    r"pl(?P<pred_len>\d+)"
    # skip any extra tokens (e.g. dm512_nh8_…)
    r"(?:_[^_]+)*"
    # finally the test_<itr> instead of _itr_
    r"_test_(?P<itr>\d+)"
)

# outside parse_log, next to MET_RE etc.
PLAIN_MET_RE2 = re.compile(
    r"\bmse:(?P<mse>[0-9.]+),\s*"
    r"mae:(?P<mae>[0-9.]+),\s*"
    r"rmse:(?P<rmse>[0-9.]+),\s*"
    r"mape:(?P<mape>[0-9.]+),\s*"
    r"mspe:(?P<mspe>[0-9.]+)"
)
PLAIN_MET_RE = re.compile(
    r"\bmse:(?P<mse>[0-9.]+),\s*"
    r"mae:(?P<mae>[0-9.]+),\s*"
)


def _convert(v):
    """Convert 'None'→None, ints, floats, or leave string."""
    if v in ('None', ''):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except:
            pass
    return v

def parse_log(path, model=None):
    rows = []
    current = {}  # holds args for the current itr
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            # 1) Did we hit a "start training" line?
            if model: m = START_RE2.search(line)
            else: m = START_RE.search(line)
            if m:
                current = {
                    'model':       m.group('model') if model is None else model,
                    'data':        m.group('data'),
                    'seq_len':     int(m.group('seq_len')),
                    'label_len':   int(m.group('label_len')),
                    'pred_len':    int(m.group('pred_len')),
                    'itr':         int(m.group('itr')),
                }
                continue

            if model: m2 = PLAIN_MET_RE2.search(line)
            else: m2 = PLAIN_MET_RE.search(line)
            if m2 and current:
                # build row from `current` + these five metrics
                row = dict(current)
                for k,v in m2.groupdict().items():
                    row[k] = float(v)
                rows.append(row)
                continue

    return rows

def main():
    p = argparse.ArgumentParser(
        description="Extract results from logs/{model}/run.log → CSV"
    )
    p.add_argument('--logs_dir', default='logs',
                   help="root folder (with subfolders named per-model)")
    p.add_argument('--output',   default='results.csv',
                   help="where to write the CSV")
    args = p.parse_args()

    all_rows = []
    for mdl in os.listdir(args.logs_dir):
        logp = os.path.join(args.logs_dir, mdl, 'run.log')
        if os.path.isfile(logp):
            if mdl.startswith('ForecastPFN'):
                print(mdl)
                all_rows.extend(parse_log(logp, model=mdl))
            else:
                print(mdl)
                all_rows.extend(parse_log(logp))

    if not all_rows:
        print("❌ No data found under", args.logs_dir)
        return

    # final column order
    headers = [
      'model','data','seq_len','label_len','pred_len','itr',
      'mae','mse','rmse','mape','mspe'
    ]

    with open(args.output, 'w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=headers)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"✔️  Wrote {len(all_rows)} rows to {args.output}")

if __name__=='__main__':
    main()
