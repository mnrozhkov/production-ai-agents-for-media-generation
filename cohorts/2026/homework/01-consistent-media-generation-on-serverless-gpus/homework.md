## Homework 1: GPU Media Worker

> [!NOTE]
> You need a Nebius project with credits. See [Module 1](../../../../01-consistent-media-generation-on-serverless-gpus/README.md).
> Stop or delete every Endpoint, Job and JupyterLab when you're done. A forgotten Endpoint bills while it's RUNNING.

In this homework, you generate one media asset on a GPU, measure it, and decide when self-hosting beats a per-call API.

### Find a creative idea

Start with what you want to create, then pick the model. A few ideas:

* A cartoon character or a comic panel
* A birthday or holiday gift card
* A product ad shot or a short product clip
* A children's storybook illustration
* A short looping video for social media
* A personal voice greeting or a podcast intro (speech)
* A jingle or background track (music)

You will reuse the same idea in Homework 3, so pick one you like.

### Setup

* Use a Jupyter notebook or a `.py` script.
* Pick **one** model from the list below. Any modality: image, video, speech or music.
* Run it any way from Module 1: an Endpoint called from your laptop, a Job, or JupyterLab on a GPU.

| Task | Model |
|---|---|
| Text-to-image | Sana, Qwen-Image, Z-Image-Turbo |
| Image editing | Qwen-Image-Edit-2511 |
| Text-to-video | Wan2.1-T2V-1.3B |
| Image-to-video | Wan2.2-I2V-A14B |
| Text-to-speech | Kokoro-82M, Chatterbox |
| Text-to-music | ACE-Step 1.5 |

Templates for all of them: [Serverless AI Cookbook](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/README.md).

### Generate and measure

See [1.5 Measuring Startup, Load and Generation Costs](../../../../01-consistent-media-generation-on-serverless-gpus/05-measuring-costs.md).

* Turn your creative idea into a prompt.
* Generate the asset **at least 3 times** with the same prompt, seed and parameters. Small differences between runs can happen; see 1.2.
* Record, in seconds:
    * **Startup time `t_start`:** from create (Endpoint, Job or JupyterLab) until it is RUNNING
    * **Ready time `t_ready`:** Endpoint: from RUNNING until `/v1/models` returns JSON (readiness wait). Job or JupyterLab: the model load timer in your code.
    * **Generation time `t_gen`:** per asset. Endpoint: request latency from your client. Job or JupyterLab: the generation timer in your code. Use the median of your runs.
* Note the GPU you ran on and its price per hour. Rates: [Nebius pricing](https://docs.nebius.com/compute/resources/pricing).
* Save the asset. If you ran a Job, it is in your bucket.

You need `t_start` and `t_ready` for the breakeven below. Keep your numbers: you will compare them with Homework 3.

### Find the API price

Find a service that sells the same model (or the closest equivalent) as a per-call API. Write down the service, the price per asset for your settings (resolution, duration or characters), and the link to its pricing page.

### Compute the breakeven

Let:

* `rate` = your GPU price per second (price per hour / 3600)
* `t_start`, `t_ready`, `t_gen` = your measured times, in seconds
* `p` = the API price per asset

Then:

* Marginal compute cost per asset: `c = rate × t_gen`
* Fixed cost of one run (startup + ready): `F = rate × (t_start + t_ready)`
* Total run cost per asset for `N` assets: `F / N + c`
* Breakeven, in assets per run: `N = ceil(F / (p − c))`

The formulas assume one startup per run, assets generated one after another, and no idle time. If `p ≤ c`, self-hosting never breaks even on this model. Enter `0`.

### Question 1

What is the creative idea behind your asset? Why did you choose this topic? Two or three sentences.

### Question 2

Which model did you use?

- Sana
- Qwen-Image
- Z-Image-Turbo
- Qwen-Image-Edit-2511
- Wan2.1-T2V-1.3B
- Wan2.2-I2V-A14B
- Kokoro-82M
- Chatterbox
- ACE-Step 1.5

### Question 3

How did you run it?

- Endpoint, called from my laptop
- Job
- JupyterLab on a GPU

### Question 4

Which GPU did you run on?

- L40S
- H100
- Other

### Question 5

What was your generation time per asset (median), in seconds?

### Question 6

What is your marginal compute cost per asset, `c`, in USD?

### Question 7

Which per-call API did you compare with, and what does it charge per asset? Format: `service, $price per asset, link`.

### Question 8

What is your self-host breakeven `N`, in number of assets?

## Learning in Public

Share your asset and the idea behind it in a short post. Use `#[placeholder]`, tag DataTalksClub and Nebius, and submit the post link with your homework. Don't share credentials, Endpoint tokens or bucket keys.

See the [learning in public guide](https://datatalks.club/docs/courses/zoomcamp-logistics/learning-in-public/) for more ideas.

### Example post for LinkedIn

```text
Completed Module 1 of Production AI Agents for Generative Media!

I ran <model> on Nebius Serverless AI and generated this <image/video/audio> (attached).
It takes <Z> s on average to generate one asset (tested on <N> samples).

The idea: <what you wanted to create and why you chose this topic>.

My code: <LINK>

#[placeholder] @DataTalksClub @Nebius
```

## Submit the results

* Submit your results here: [placeholder]
* All questions are about your own run. Any honest answer counts.
