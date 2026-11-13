<p align="center">
  <img src="../images/module-4-cover.png" alt="Module 4, Make it production-ready: Evals, Tracing and SLOs. System, agent and media evals, multimodal judges, cost and time per step, SLO, dashboard and alert. You ship the capstone brief" width="100%">
</p>

# Module 4: Evaluation, Observability and Production

Welcome to Module 4 of Production AI Agents for Generative Media! This week we make it production-ready: measure how good your assets are, explain what each record cost and why it took that long, and decide whether the factory is good enough to ship.

> [!NOTE]
> This module builds on Modules 1 to 3: the measurements and cost terms from [1.5](../01-consistent-media-generation-on-serverless-gpus/05-measuring-costs.md), the judge and rubric v1 from [2.5](../02-multi-agent-media-workflows/05-judge-loop.md), the budget and quality thresholds from [3.4](../03-batch-media-generation-at-scale/04-breaking-the-pipeline.md), and your Homework 3 run folder (`events.jsonl`, `metrics.json`, `summary.json`, `items/`, `dead_letter.jsonl`). Every lesson works on that run folder. If you skipped Homework 3, use the reference run published on Mon 14 Dec.

> [!WARNING]
> Most of this module reads an existing run folder and needs no GPU. Two things in 4.2 are live. The candidate run: keep it to the 10 golden records, set a `--timeout` on every Job, and stop your Endpoints when you finish. Re-judging under rubric v2: vision-model calls on the disagreements and held-out items only, at most 40 images. Everything else uses the judge verdicts and transcripts already stored in the run folder, so you don't pay for the judge twice.

* Video Link (Coming Soon)
* Slides (Coming Soon)

<p align="center">
  <img src="../images/diagram-evals-tracing-slo.png" alt="Module 4: evals at the system, agent and media layers, a trace with cost and time per step, and an SLO with a dashboard and an alert" width="100%">
</p>

## What you will learn

By the end of this module, you will be able to:

* Choose metrics for the system, agent and media layers and compute them from a run folder, cheap deterministic checks first
* Calibrate the 2.5 judge against your own labels, fix the rubric where they disagree, and say how far you trust it
* Block a change to a model, prompt or configuration with a golden set and a regression gate
* Answer "why did this record cost $X and take N s?" from one trace
* Sketch a production architecture for a media agent, set an SLO with an alert on burn rate, and make a go/no-go call
* Turn auth, quotas, abuse, prompt injection through tools, content safety, licensing and provenance into stated controls in your capstone brief

Default path for the whole module: your Homework 3 run folder as input, the 2.5 judge with a pinned model as the judge, a pinned [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) model for the speech check, and `events.jsonl` as the trace. The reference run ships cached judge verdicts and Whisper transcripts for every asset, so 4.1 and the rubric v1 calibration in 4.2 run offline. The tracing backend is to be announced; everything in this module works from the JSONL files alone. Each lesson writes its output to `runs/04-evaluation-observability-production/<run_id>/` and leaves your Homework 3 run folder unchanged: `evals.json` and `labels.csv` (4.1), `calibration.json` and `gate.json` (4.2), `waterfall.md` (4.3), `slo.md` and `dashboard.html` (4.4), `secrets_scan.json` (4.5). 4.5 also writes into your capstone brief.

## Lessons

* [4.1 Evaluating Generative Systems: System, Agent and Media](01-evaluating-generative-systems.md)
* [4.2 Judge Calibration and Regression Gates](02-multimodal-judges.md)
* [4.3 Observability and Tracing](03-observability-tracing.md)
* [4.4 Designing for Production: Architecture, SLO, Dashboard and Alerts](04-designing-for-production.md)
* [4.5 Production Concerns and the Capstone Brief](05-production-concerns.md)

## Capstone Step

Module 4 has no homework. Apply it in your capstone: an eval suite at the system, agent and media layers with a golden set and a regression gate (4.1, 4.2), one trace per record (4.3), an SLO with a dashboard and your go/no-go call (4.4), and one stated control per production concern, a licence note per model and a provenance statement per asset (4.5). The capstone brief is published with this module. See [projects](../projects).

## Module structure (proposed)

One page per lesson at the module root, all code flat in `code/`, images in `images/`.

```text
04-evaluation-observability-production/
├── README.md                       # lessons index
├── 01-evaluating-generative-systems.md
├── 02-multimodal-judges.md
├── 03-observability-tracing.md
├── 04-designing-for-production.md
├── 05-production-concerns.md
├── images/
└── code/
    ├── README.md                   # how to run each file, env vars, run folder paths
    ├── models.yaml                 # judge model and Whisper model, pinned
    ├── runfolder.py                # read a run folder from a local path or s3:// (boto3, endpoint_url)
    ├── checks.py                   # 4.1: image, video, audio checks; ASR name check
    ├── evals.py                    # 4.1: run folder → evals.json
    ├── rubric.yaml                 # 4.2: rubric v2, starts from the 2.5 v1; criteria with pass and fail examples, version
    ├── calibrate.py                # 4.2: labels.csv + ledger verdicts → calibration.json
    ├── golden.csv                  # 4.2: the 10 golden record IDs, seeds, configuration
    ├── gate.py                     # 4.2: baseline vs candidate → gate.json, exit 1 on fail
    ├── eval-gate.yml               # 4.2 also try: GitHub Actions workflow on the golden set
    ├── waterfall.py                # 4.3: run folder + record ID → waterfall.md
    ├── budget_agent.py             # 4.3 also try: reserve / commit / release, typed refusals
    ├── slo.py                      # 4.4: SLIs, error budget, burn rate → slo.md section; --check exits 1 on burn
    ├── dashboard.py                # 4.4: summary.json + evals.json → dashboard.html
    ├── capstone_brief.md           # 4.5: brief template; one control per concern, licences, provenance
    ├── licences.md                 # 4.5: licence note per course model [TBC]
    ├── scan_secrets.py             # 4.5: run folder + stored results → secrets_scan.json, exit 1 on a hit
    └── fixtures/
        └── run_10/                 # 10 records cut from the reference run; every script runs on it offline
```

# Community Notes

Did you take notes? You can share them here:

* Add your notes here!
