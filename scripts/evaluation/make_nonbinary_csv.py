from pathlib import Path
import csv, os

BASE = str(Path(__file__).resolve().parents[2])
INPUT  = os.path.join(BASE, 'data', 'eval_results', 'annotated_350+350_TP_TN_FP_FN.csv')
OUTPUT = os.path.join(BASE, 'data', 'eval_results', 'annotated_nonbinary_full.csv')

def confidence_tier(s):
    pct = round(s * 100)
    if pct <= 30:
        return 'High Confidence Authentic (0%-30%)'
    elif pct < 50:
        return 'Moderately Confident Authentic (31%-49%)'
    elif pct == 50:
        return 'Inconclusive / Neutral (50%)'
    elif pct < 70:
        return 'Moderately Confident Deepfake (51%-69%)'
    else:
        return 'High Confidence Deepfake (70%-100%)'

rows = list(csv.DictReader(open(INPUT, encoding='utf-8')))

fields = ['row_no', 'clip_id', 'fake_label', 'method', 'type',
          'raw_score', 'confidence_tier', 'pred_cal', 'annotation']

with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        s = float(r['raw_score'])
        w.writerow({
            'row_no'         : r['row_no'],
            'clip_id'        : r['clip_id'],
            'fake_label'     : r['fake_label'],
            'method'         : r['method'],
            'type'           : r['type'],
            'raw_score'      : r['raw_score'],
            'confidence_tier': confidence_tier(s),
            'pred_cal'       : r['pred_cal'],
            'annotation'     : r['annotation'],
        })

print(f'Saved {len(rows)} rows -> {OUTPUT}')
print()

# Preview first 15 rows
rows2 = list(csv.DictReader(open(OUTPUT, encoding='utf-8')))
print(f"{'#':>3}  {'raw_score':>10}  {'confidence_tier':<42}  {'pred':>4}  {'annot'}")
print("-" * 80)
for r in rows2[:15]:
    print(f"{r['row_no']:>3}  {float(r['raw_score']):>10.4f}  {r['confidence_tier']:<42}  {r['pred_cal']:>4}  {r['annotation']}")

# Tier counts & accuracy breakdown
from collections import defaultdict
order = [
    'High Confidence Authentic (0%-30%)',
    'Moderately Confident Authentic (31%-49%)',
    'Inconclusive / Neutral (50%)',
    'Moderately Confident Deepfake (51%-69%)',
    'High Confidence Deepfake (70%-100%)',
]
tier_stats = defaultdict(lambda: {'total': 0, 'real': 0, 'fake': 0, 'correct': 0})
for r in rows2:
    t = r['confidence_tier']
    fl = int(r['fake_label'])
    tier_stats[t]['total'] += 1
    if fl == 0:
        tier_stats[t]['real'] += 1
    else:
        tier_stats[t]['fake'] += 1
    if 'Authentic' in t and fl == 0:
        tier_stats[t]['correct'] += 1
    elif 'Deepfake' in t and fl == 1:
        tier_stats[t]['correct'] += 1

print()
print("=== 5-Tier Continuous Confidence Scale Breakdown (N = 700) ===")
print(f"{'Tier':<42} {'Count':>6} {'% Total':>8} {'Real':>5} {'Fake':>5} {'Tier Accuracy':>14}")
print("-" * 84)
for t in order:
    st = tier_stats[t]
    pct = (st['total'] / len(rows2)) * 100
    acc = (st['correct'] / st['total'] * 100) if st['total'] > 0 else 0.0
    print(f"{t:<42} {st['total']:>6} {pct:>7.2f}% {st['real']:>5} {st['fake']:>5} {acc:>13.2f}%")