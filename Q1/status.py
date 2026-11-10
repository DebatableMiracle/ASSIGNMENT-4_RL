
"""python status.py -- progress / latest eval of every run in results/."""
import glob, json, os
 
print(f"{'run':<22}{'progress':>9}{'eta':>10}{'last eval succ':>16}{'final succ':>12}")
for d in sorted(glob.glob("results/*/")):
    name, p, fin = os.path.basename(d.rstrip("/")), d + "progress.json", d + "final.json"
    if os.path.exists(fin):
        print(f"{name:<22}{'100.0%':>9}{'-':>10}{'-':>16}{json.load(open(fin))['success_rate']:>12.2f}"); continue
    if os.path.exists(p):
        j = json.load(open(p)); print(f"{name:<22}{j['pct']:>8.1f}%{j['eta_s'] / 60:>8.0f}m{j['last_eval_succ']:>16.2f}{'-':>12}")
    else: print(f"{name:<22}{'starting':>9}")
 
