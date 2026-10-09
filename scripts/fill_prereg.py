"""Fill scripts/prereg_template.md with the P2 numbers -> pre_registered_hypotheses.md (then commit it).
Reads only results/metrics/p2_summary.json (counts, class rates, power) - no hypothesis outcomes."""
import datetime as dt
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p2 = json.loads((root / "results/metrics/p2_summary.json").read_text())
c = p2["tier1_eligible_counts"]
rows = ["| Family | train | validation | test |", "|---|---|---|---|"]
for fam in ["strict", "broad", "all"]:
    d = c.get(fam, {})
    rows.append(f"| {fam.upper()} | {d.get('train', 0):,} | {d.get('validation', 0):,} | {d.get('test', 0):,} |")
cr = "; ".join(f"{fam.upper()} 10d {v['label_10']:.3f} / 20d {v['label_20']:.3f} / 60d {v['label_60']:.3f}"
               for fam, v in p2["class_balance"].items())
pw = ["| Sample | n | base rate | H1 MDE (pp) | H2 MDE (pp) | H4 MDE (AUC) |", "|---|---|---|---|---|---|"]
for k, v in p2["power"].items():
    pw.append(f"| {k} | {v.get('n', 0):,} | {v.get('base_rate', '')} | {v.get('H1_mde_pp', '')} | {v.get('H2_mde_pp', '')} | {v.get('H4_mde_auc', '')} |")
t = (root / "scripts/prereg_template.md").read_text(encoding="utf-8")
t = (t.replace("{REG_DATE}", dt.date.today().isoformat()).replace("{COUNTS_TABLE}", "\n".join(rows))
      .replace("{CLASS_RATES}", cr).replace("{POWER_TABLE}", "\n\n" + "\n".join(pw))
      .replace("{POWER_TRIGGERED}", "yes" if p2["power_rule_triggered"] else "no"))
(root / "pre_registered_hypotheses.md").write_text(t, encoding="utf-8")
print("written pre_registered_hypotheses.md")
