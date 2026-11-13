## Homework 3: Factory Scale Challenge

> [!NOTE]
> You need a Nebius project with credits and a quota of at least 10 preemptible VMs, 10 disks and the matching allocations in your region. A new account starts with 2. Request the increase before you start. See [Module 3](../../../../03-batch-media-generation-at-scale/README.md).
> Cancel every Job and stop every Endpoint when you're done.

In this homework, you scale the Personalized Media Factory to 200 records over 10 Jobs in parallel on spot GPUs, survive a kill, and report what one accepted asset costs.

### Setup

* Records: all 200 records in [`examples/mediafactory/data/kids.csv`](../../../../examples/mediafactory/data/kids.csv) (`id`, `name`, `age`, `wishlist`)
* Prompts: your Homework 1 creative idea, personalized per record
* Assets per record: card text, image and voice. Video for a 10-record sample only, picked with a fixed seed.
* Models (fixed for the leaderboard): Sana for images, Kokoro-82M for voice, Wan2.2-I2V-A14B for the video sample
* Jobs: 10 spot Jobs in parallel (`--preemptible --follows-spot-price`), each with a `--timeout`
* Your configuration from [3.3](../../../../03-batch-media-generation-at-scale/03-optimizing-cost.md)
* A `budget_usd` and a judge pass-rate floor on the run, as in [3.4](../../../../03-batch-media-generation-at-scale/04-breaking-the-pipeline.md)

### Run

* Before you launch, predict the wall time and the total run cost with the [3.1](../../../../03-batch-media-generation-at-scale/01-fan-out.md) formulas
* Launch the run. At about 50%, cancel at least one Job with `nebius ai job cancel`, then resume with the same `run_id`
* Take the spot and on-demand rates from [Nebius pricing](https://docs.nebius.com/compute/resources/pricing) and write down the date

The formulas assume each Job pays its own startup and model load, assets inside a Job run one after another, and no idle gaps.

### Question 1

What wall time did you predict for the run, and what did you get? Format: `predicted min, actual min`.

### Question 2

How many Jobs ran in parallel at peak?

- 10
- 5–9
- Fewer than 5

### Question 3

What was the total cost of the run, in USD?

### Question 4

What was your cost per accepted asset, in USD?

### Question 5

How much did spot save compared with on-demand for the same billable time, in USD?

### Question 6

How many items that were done before the kill were regenerated after the resume?

### Question 7

Which 3.3 change lowered your total run cost per asset the most?

- Cached weights (lower `F`)
- bf16, attention backend or batch size (lower `t_gen`)
- `torch.compile`
- Fewer steps
- Offloading or lower precision to fit a smaller GPU
- Other

### Question 8

What were your p50 and p95 generation times per image, in seconds? Format: `p50, p95`.

### Question 9

What was the judge pass rate on your run, in percent?

### Question 10

If you used Sana or Kokoro-82M in Homework 1: what was your median generation time then, and what is it now, in seconds? Format: `HW1 s, HW3 s`. Otherwise enter `n/a`.

## Learning in Public

Share your run in a short post. Use `#[placeholder]`, tag DataTalksClub and Nebius, and submit the post link with your homework. Don't share credentials, Endpoint tokens or bucket keys.

See the [learning in public guide](https://datatalks.club/docs/courses/zoomcamp-logistics/learning-in-public/) for more ideas.

### Example post for LinkedIn

```text
Completed Module 3 of Production AI Agents for Generative Media!

I generated personalized <image + voice> assets for <N> records on Nebius Serverless AI, over 10 spot GPU Jobs in parallel.
One accepted asset cost $<X>. Spot saved $<Y> vs on-demand.
I killed the run halfway and resumed it: <Z> finished items were regenerated.
The change that mattered most: <your 3.3 change>.

My code: <LINK>

#[placeholder] @DataTalksClub @Nebius
```

## Submit the results

* Submit your results and your `summary.json` here: [placeholder]
* All questions are about your own run. Any honest answer counts.
