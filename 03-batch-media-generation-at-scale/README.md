<p align="center">
  <img src="../images/module-3-cover.png" alt="Module 3, Make it scale: Batch Media Generation at Scale. Fan-out on spot GPUs, per-item checkpoints, breaking the pipeline on purpose, budget-aware agents, quality vs cost. You ship Homework 3: Factory Scale Challenge" width="100%">
</p>

# Module 3: Batch Media Generation at Scale

Welcome to Module 3 of Production AI Agents for Generative Media! This week we make it scale: run the same generation for every record over parallel spot GPUs, survive a killed run without paying twice, cut the cost per asset, and stop the run when it breaks a budget or a quality bar.

> [!NOTE]
> This module builds on Module 1 and Module 2: Jobs, the Object Storage bucket and the cost terms from [1.4](../01-consistent-media-generation-on-serverless-gpus/04-serving-vs-batch.md) and [1.5](../01-consistent-media-generation-on-serverless-gpus/05-measuring-costs.md), the idempotency key from [2.2](../02-multi-agent-media-workflows/02-tools.md), tickets from [2.4](../02-multi-agent-media-workflows/04-long-running-tools.md) and the judge and rubric v1 from [2.5](../02-multi-agent-media-workflows/05-judge-loop.md).

> [!IMPORTANT]
> Parallel Jobs need quota. Serverless AI Jobs count toward Compute quotas: one VM, one disk and one allocation per running Job, plus GPUs for regular VMs. A new account starts with 2 regular and 2 preemptible VMs per region. Homework 3 runs 10 Jobs in parallel, so request an increase to at least 10 preemptible VMs, 10 disks and the matching allocations before the homework week. How to check and increase quotas: [3.1](01-fan-out.md#quotas-set-your-k).

> [!WARNING]
> Cancel your Jobs (`nebius ai job cancel <job_ID>`) and stop your Endpoints when you finish the module to avoid charges. Set a `--timeout` on every Job and a `budget_usd` on every run. Run the lessons on 10 to 100 records; all 200 records is the homework. Keep video to a 10-record sample.

* Video Link (Coming Soon)
* Slides (Coming Soon)

<p align="center">
  <img src="../images/diagram-batch-scale.png" alt="1,000 records fan out to spot GPU workers with a per-item checkpoint. A preempted worker returns its items to the queue. Each result is written once to object storage, with a cost report. A killed run resumes and skips items that are already done" width="100%">
</p>

## What you will learn

By the end of this module, you will be able to:

* Predict the wall time and cost of a batch run before you launch it, then fan out records over parallel Jobs and fan the results back in
* Run batch work on spot (preemptible) GPUs, checkpoint every item, and prove that a killed run resumes without regenerating finished items
* Choose a chunk size from your own fixed cost, marginal cost and preemption count
* Lower the cost per asset with three levers: fixed cost (cold start), generation time, and the GPU you need (VRAM)
* Break the pipeline on purpose, including budget and quality thresholds, and fix each failure with a named pattern
* Optional: fine-tune an image model on a new style as a Job and use it in your pipeline

Default path for the whole module: [Sana](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-sana/README.md) on L40S for images, [Kokoro-82M](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-kokoro-82m/README.md) for voice and [Wan2.2-I2V-A14B](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-wan22-i2v-a14b/README.md) on H100 for the video sample. Records: [`examples/mediafactory/data/kids.csv`](../examples/mediafactory/data/kids.csv), 200 records (`id`, `name`, `age`, `wishlist`), and its first 10 rows in [`kids_10.csv`](../examples/mediafactory/data/kids_10.csv). Lessons that use 100 records take the first 100 rows of `kids.csv`. Every run writes `events.jsonl`, `metrics.json` and `summary.json` to its run folder in the bucket.

## Lessons

* [3.1 One to 1,000: Fan-out with Jobs](01-fan-out.md)
* [3.2 Spot GPUs and Design for Interruption](02-spot-and-interruption.md)
* [3.3 Optimizing Cost per Asset: Cold Start, VRAM and Speed](03-optimizing-cost.md)
* [3.4 Breaking the Pipeline on Purpose](04-breaking-the-pipeline.md)
* [3.6 Fine-Tune a Model on a New Style (Optional, Advanced)](06-fine-tune-style.md)

## Homework 3: Factory Scale Challenge

Scale the Personalized Media Factory to all 200 records in `kids.csv` over 10 Jobs in parallel: card text, image and voice for every record, video for a 10-record sample. Use spot Jobs, your 3.3 configuration, and survive at least one kill → resume. Compete on the leaderboard: the cheapest reliable run at acceptable quality wins.

* Before you start: a quota of at least 10 preemptible VMs (see the note at the top)
* Submit your `summary.json`: records, Jobs, completed, dead-lettered, total cost, cost per accepted asset, saved vs on-demand, p50 and p95 time per asset, judge pass rate, and the resume proof (items done before the kill, items regenerated after it)
* Leaderboard: ranked by cost per accepted asset, among runs that meet the completion and quality bars

Questions and submission: [homework.md](../cohorts/2026/homework/03-batch-media-generation-at-scale/homework.md).

## Capstone Step

Run your own records through your capstone pipeline on 100 records (10 for video) with one kill → resume. Keep `summary.json`: it is your capstone cost report. See [projects](../projects).

## Module structure (proposed)

One page per lesson at the module root, all code flat in `code/`, images in `images/`.

```text
03-batch-media-generation-at-scale/
├── README.md                       # lessons index
├── 01-fan-out.md
├── 02-spot-and-interruption.md
├── 03-optimizing-cost.md
├── 04-breaking-the-pipeline.md
├── 06-fine-tune-style.md
├── images/
└── code/
    ├── README.md                   # how to run each file, env vars, quota check
    ├── predict.py                  # 3.1: metrics.json + N + K → wall time, total cost
    ├── worker.py                   # 3.1–3.4: runs in a Job; chunk → assets + ledger; FAULT switch
    ├── launch.py                   # 3.1–3.2: manifest → chunks → K Jobs → wait → resubmit → summary.json
    ├── ledger.py                   # 3.2: items/<id>.json read/write, skip if done, dead-letter; stores verdict, rubric_version, judge_model, reasons
    ├── kill_resume.sh              # 3.2: start, cancel at ~50%, resume, print diff
    ├── chunk_sweep.py              # 3.2 also try: chunk sizes 5, 10, 20 on 40 records
    ├── sync_weights.py             # 3.3: HF → bucket, pinned revision
    ├── bench.py                    # 3.3: one config per Job, writes metrics.json
    ├── compare.py                  # 3.3: metrics.json files → table, total run cost per asset at N
    ├── chaos.py                    # 3.4: one fault per run on 10 records, expected vs actual
    ├── prepare_dataset.py          # 3.6: dedupe, VLM captions, flag style words, contact sheet
    ├── lora_config.yaml            # 3.6: AI Toolkit config
    ├── train_lora_job.sh           # 3.6: nebius ai job create … --timeout
    └── finetune_tool.py            # 3.6: start_finetune / get_status, 2.4 ticket pattern
```

Shared records at the repo root:

```text
examples/mediafactory/data/
├── kids.csv                        # 200 records. Homework 3 and 100-record lessons (first 100)
├── kids_10.csv                     # first 10 rows of kids.csv
└── kids_bad.csv                    # bad records for 3.4
```

# Community Notes

Did you take notes? You can share them here:

* Add your notes here!
