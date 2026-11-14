---
video_url: ""
code:
  - label: Cost per asset (notebook)
    path: code/nb-measuring-costs.ipynb
  - label: Generation settings experiments (notebook)
    path: code/generation_settings.ipynb
  - label: Record a Job's lifecycle times
    path: code/measure_job.py
  - label: Record an Endpoint run's times
    path: code/measure_endpoint.py
prev_url: 04-serving-vs-batch.md
next_url: ../02-multi-agent-media-workflows/01-agent-loop.md
---
# 05 - Measuring Startup, Load and Generation Costs

<!-- Draft v1, 2026-11-12: written from the v1 code. [measure] values come from the author runs. -->

Say your first image took a few minutes from **Create** to the file on your laptop. So what does one image cost? There's no single answer: startup and model load are paid once per run, generation once per asset, and idle time for every minute you forget to stop. In this lesson we split one run into those parts, turn them into a cost per asset at the regular price and at today's preemptible price, and compare it with a per-call API. Then we see which generation settings cost time and which only change the picture.

You're done when you have:

- the cost per asset at N = 1, 10 and 100, at the regular price, with and without 10 minutes of idle time;
- the same at today's preemptible price, and the difference;
- a breakeven N against one per-call API price, with its link;
- one settings comparison: the images, the time per image, and a line on what changed.

Time and cost: about 30 minutes. With our sample run the cost notebook needs no GPU and no credits. The settings experiments need a running Sana Endpoint for a few minutes `[measure]`.

## Four measurements

| Field in `metrics.json` | What it is | Where it comes from |
| --- | --- | --- |
| `startup_s` | Create → `RUNNING` | Watching the status |
| `ready_wait_s` (Endpoint) or `load_s` (Job) | `RUNNING` → first answer; or model load in our worker | Polling `/v1/models`; or a timer in the worker |
| `latency_s` | Generation time per asset; we use the median | Timer around each request or generation |
| `billable_s` | Create → stopped or finished; includes idle time | Watching the status |

`[placeholder: the Job's load_s and latency_s. The image-only worker generate.py was removed on 2026-11-13. The lesson 04 worker prints the load time and the time per asset for every step in the Job's logs and writes no metrics.json; decide how this lesson reads them]`

The docs give the current status (`status.state`) but no timestamp for each change, so we record when we see each change ourselves. <!-- c:M1-339 --> For a Job, start this right after creating it:

```bash
uv run python code/measure_job.py --watch <job_ID>
```

It prints each status change, then the three times to attach to the Job's `metrics.json` after you download it:

```bash
uv run python code/measure_job.py runs/.../metrics.json \
  --created-at <...> --running-at <...> --finished-at <...>
```

For an Endpoint, [code/measure_endpoint.py](code/measure_endpoint.py) runs the same config as the Job and takes the times you note in the console (`--created-at`, `--running-at`).

The times are as accurate as the polling interval: a few seconds. `billable_s` is our estimate of the billed time, not the invoice. Billing is per second while running. [Serverless overview](https://docs.nebius.com/serverless/overview), [Compute pricing](https://docs.nebius.com/compute/resources/pricing) <!-- c:M1-337 c:M1-338 -->

## From times to cost

- Marginal cost per asset: `c = rate × t_gen`, with `t_gen` the median of `latency_s`.
- Fixed cost per run: `F = rate × (t_start + t_ready)`. Idle time adds `rate × t_idle`.
- Total per asset for N assets: `F / N + c`.
- Breakeven against an API at `p` per asset: `N = ceil(F / (p − c))`, or `0` if `p ≤ c`.

**Assumptions:** one startup per run, assets generated one after another, no idle time unless you add it.

**Left out:** failed or retried generations, engineering and ops time, storage and egress, and quality differences against the API. Each one makes self-hosting look cheaper than it is.

`c` is the marginal cost, never "the cost per asset": with N = 1, the fixed part dominates.

## Run the cost notebook

Open [code/nb-measuring-costs.ipynb](code/nb-measuring-costs.ipynb).

1. **Load one run.** Our sample run, or your own `metrics.json`. `[placeholder: sample runs from the author run]`
2. **Regular price (the default).** Type the regular price per hour for the run's GPU and preset from [Compute pricing](https://docs.nebius.com/compute/resources/pricing). L40S lists GPU, vCPU and RAM separately: for `1gpu-8vcpu-32gb`, add GPU + 8 × vCPU + 32 × RAM. H100 and H200 list one price per GPU hour. <!-- c:M1-345 -->
3. **Cost per asset** at N = 1, 10 and 100, then again with 10 minutes of idle time.
4. **Today's preemptible price.** For GPUs with a spot price, the notebook reads it from the public [spot-price API](https://docs.nebius.com/signup-billing/spot-prices) (no login). It can change every 15 minutes, so the notebook prints when it read it. L40S isn't listed: its preemptible price is fixed on the pricing page. <!-- c:M1-344 c:M1-346 -->
5. **Breakeven.** Find a per-call API for the same model, or the closest one, and type its price per asset for your settings and the link. Units differ by provider (per megapixel, per video second, per 1,000 characters): convert to per asset first. For Sana, fal sells it per megapixel. [fal: Sana](https://fal.ai/models/fal-ai/sana) <!-- c:M1-347 -->

Our worked example, at regular prices `[placeholder: author runs on L40S; rate and date in the caption]`:

| Run | t_start | t_ready | t_gen | c (marginal) | per asset at N=1 / 10 / 100 |
| --- | --- | --- | --- | --- | --- |
| Endpoint | `[measure]` | `[measure]` | `[measure]` | `[calc]` | `[calc]` |
| Job | `[measure]` | `[measure]` | `[measure]` | `[calc]` | `[calc]` |

Spot isn't always cheaper in the same way every hour; the preemptible price is capped just below the regular rate. <!-- c:M1-345 -->

## Generation settings: time vs pixels

Some settings make the model do more work per asset, so they cost time and money. Others only change the result. You need both kinds written down to make the same asset again.

**Image (Sana).** These values travel in the POST body to `/v1/images/generations`. With the same prompt and seed, changing any one of them produces a different image. The defaults are 20 steps, guidance `4.5` and an empty negative prompt, matching `generation.json`. <!-- c:M1-211 -->

- `num_inference_steps` is how many denoising steps the model takes. It starts from noise and removes a little at each step. 20 is enough for this Sana checkpoint to finish the picture. Fewer steps leave it softer and less settled. More steps refine it further, and each step adds time: 40 steps take about `[measure]` as long as 20. Past a point the picture stops changing in a useful way.
- `guidance_scale` is how hard the model is pushed toward the prompt. At each step it predicts the prompt and an empty prompt, then steps farther toward the prompt by this factor. `4.5` is Sana's usual setting. Lower values let the model wander, so the scene matches the words less closely. Higher values lock onto the words and tend to look harsher, with blown contrast and odd details. Whether `1.0` also saves time `[verify]`.
- `negative_prompt` is the text used for that "away" prediction. An empty string, the default, means "away from nothing in particular". A value such as `"text, watermark, blurry"` steers the image away from those things. It barely changes the time `[measure]`. It does change the pixels, so it belongs with the seed and the other settings when you want the same image again.
- `size`: more pixels, more work per step `[measure: 512² vs 1024²]`.

Run [code/generation_settings.ipynb](code/generation_settings.ipynb) against a running Sana Endpoint: each cell changes one setting and shows the images with the time each took. Stop the Endpoint at the end.

`[placeholder: four comparison strips from the author run, with times: steps, guidance, negative prompt, size]`

**Other modalities.** The same rule holds; the settings differ by model family (request arguments from the [cookbook templates](https://github.com/nebius/serverless-ai-cookbook/tree/main/templates)):

| Modality | Model | Costs time | Changes only the result | Notes |
| --- | --- | --- | --- | --- |
| Image | Qwen-Image | steps, size | seed | 20 steps in the template <!-- c:M1-361 --> |
| Image | Z-Image-Turbo | fixed at ~9 steps | seed | Distilled: keep ~9 steps and guidance 0; more steps make it blurry <!-- c:M1-360 --> |
| Image edit | Qwen-Image-Edit-2511 | steps, size | seed, prompt | Takes an input image <!-- c:M1-362 --> |
| Video | Wan2.1-T2V-1.3B, Wan2.2-I2V-A14B, TI2V-5B | frames, size, steps | seed, guidance `[verify]` | `flow_shift`, `boundary_ratio`, `guidance_scale_2` are family-specific; whether `fps` changes compute `[verify]` <!-- c:M1-363 c:M1-364 c:M1-325 --> |
| Speech | Kokoro-82M, Chatterbox | length of the input text | voice | No steps; Chatterbox has a guidance setting (expressiveness vs stability) <!-- c:M1-365 c:M1-366 --> |
| Music | ACE-Step 1.5 | audio duration, steps, `thinking` | seed, bpm, lyrics | `thinking: true` adds a language-model pass <!-- c:M1-367 --> |

Video and music cost scales with the length of the output, and per-call APIs price them the same way: per video second or per audio second. [fal: Wan 2.2 I2V](https://fal.ai/models/fal-ai/wan/v2.2-a14b/image-to-video), [fal: ACE-Step](https://fal.ai/models/fal-ai/ace-step) <!-- c:M1-353 c:M1-359 -->

## Check what you spent

The console shows your balance under **Billing** and the spend per project, service and resource under **Billing → Usage**. The docs describe no separate "credits remaining" view. Compare the usage for your run with `billable_s × rate`. The usage export is hourly, so it won't match to the second. [Billing: usage](https://docs.nebius.com/signup-billing/usage/view) <!-- c:M1-341 c:M1-342 c:M1-343 -->

## Common failures and recovery

| Mistake | Fix |
| --- | --- |
| A `null` measurement treated as 0 | `calculate_cost` refuses `null`; measure it or leave the run out |
| Idle time forgotten | Add `idle_s` for the minutes the Endpoint ran unused |
| A rate for the wrong GPU or preset | Use the rate for the GPU and preset the run used |
| Spot and regular prices mixed up | Label each table with the price it used |
| L40S GPU price without its vCPU and RAM lines | Add all three for the preset |
| Timestamps without a timezone | Use UTC with a `Z` or `+00:00` |
| Quoting the breakeven without what it leaves out | Keep the "left out" line next to the number |

Next: Module 2 puts an agent in front of these models.

## References

- [Getting spot prices](https://docs.nebius.com/signup-billing/spot-prices), Nebius
- [Compute pricing](https://docs.nebius.com/compute/resources/pricing), Nebius
- [ServerlessLLM (OSDI '24)](https://arxiv.org/abs/2401.14351), Fu et al.
- [How to price serverless GPUs](https://modal.com/blog/how-to-price-serverless), Modal (2026)
- [Fast cold starts for serverless GPU inference](https://www.beam.cloud/blog/cold-start-serverless-gpu-inference), Beam (2026)
- [Full Stack Deep Learning: cloud GPUs](https://fullstackdeeplearning.com/cloud-gpus/)
- [Numbers every LLM developer should know](https://github.com/ray-project/llm-numbers), Anyscale
