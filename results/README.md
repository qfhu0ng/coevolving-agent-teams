# Results

Final results of the project (concluded October 2026). The main [README](../README.md) summarises them; this
folder holds the data behind that summary. The raw run records (agent transcripts, relay events, git
histories) are not published.

| Path | Contents |
|---|---|
| `data/runs.csv` | Every scored run: task, strategy, duration, official passed/valid tests, workers spawned, known cost |
| `data/m1_rerun_pass_rate_curve.csv` | Official score of the M1 rerun's delivery every 30 minutes, replayed from its git history |
| `data/team_held_out_measurements.csv` | Held-out pass rates the two M1 teams measured on case sets they built themselves, as announced on their board |
| `data/team_size_timeline.json` | Active workers per 10 minutes and the times the Stop hook blocked the master, for the three pandoc team runs |
| `strategies/M0.md` | The hand-written starting master strategy |
| `strategies/M1.md` | The evolved strategy, the optimizer's rationale and the reviewer's listed risks |
| `figures/` | Architecture diagram and charts; `python results/figures/make_figures.py` redraws the charts from `data/` |

Notes on the numbers:

- **Official scores.** These are ProgramBench's kept-test pass counts from the official evaluator. The two
  M1 runs were scored after the run, with the same evaluator: a container that was slow to delete had made
  the runner skip their evaluation.
- **Curve.** Each curve point evaluates the last commit on the delivery branch at that minute. Work that was
  not yet committed at that moment is not counted.
- **Held-out sets.** These are the team's own proxies, not the hidden tests. The M1 rerun's set was also
  used to steer fixes, so its readings overstate the official score.
- **Costs.** Costs come from provider usage at DeepSeek's published prices. Requests that ended without
  usage data are not counted.
