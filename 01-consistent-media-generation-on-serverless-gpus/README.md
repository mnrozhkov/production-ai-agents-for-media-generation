<p align="center">
  <img src="../images/module-1-cover.png" alt="Module 1, Make it work: Consistent Media Generation on Serverless GPUs. Media model landscape, templates, containers and model cache, seeds, fitting VRAM, startup, load and generation time. You ship Homework 1: GPU Media Worker" width="100%">
</p>

# Module 1: Consistent Media Generation on Serverless GPUs

Welcome to Module 1 of Production AI Agents for Generative Media! These two weeks we make it work: run image, video and audio models on serverless GPUs, for serving and for batch, and measure what one asset costs.

> [!NOTE]
> Before you start: join the Builder Program, create a Nebius AI Cloud account with payment details and apply your promo code. Steps: see [Setup](#setup) below.

> [!WARNING]
> Make sure you stop or delete your resources (Endpoints, Jobs, JupyterLab) when you finish the module to avoid charges. An Endpoint bills while it's RUNNING, even when it serves no requests.

* Video Link (Coming Soon)
* Slides (Coming Soon)

<p align="center">
  <img src="../images/diagram-media-factory.png" alt="The Personalized Media Factory: a brief goes to an agent, GPU models generate image, voice or video, a check sends assets that aren't good enough back to revise and regenerate, and the factory scales from 1 to 1,000+ records with a cost per asset" width="100%">
</p>

## What you will learn

By the end of this module, you will be able to:

* Compare generative media modalities and model families (size, VRAM, latency, quality, licence) and explain why video is expensive
* Generate your first image, video or audio asset in minutes: through an Endpoint, or in JupyterLab on a GPU
* Explain what goes into a container for generative AI and what stays out of it (weights, secrets, outputs)
* Run the same generation for serving (Endpoint) and for batch inference (Job), and choose between them
* Measure startup, readiness and generation time, and turn them into marginal and total cost per asset

## Lessons

* [01 - Landscape: Modalities and Model Families](01-landscape.md)
* [02 - First Media Generation in a Few Minutes](02-first-generation.md)
* [03 - Containers for Generative AI](03-containers.md)
* [04 - Serving vs Batch Inference: the Same Generation for Different Use Cases](04-serving-vs-batch.md)
* [05 - Measuring Startup, Load and Generation Costs](05-measuring-costs.md)

## Models used in this module

All with 1-click templates in the [Serverless AI Cookbook](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/README.md). The default path is [Sana](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-sana/README.md) on L40S; the others are there to explore.

| Task | Model | Template GPU | Deploy |
|---|---|---|---|
| Text-to-image | [Sana](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-sana/README.md) | L40S | [Create Endpoint](https://console.nebius.com/serverless/endpoint/create?template=sana) |
| Text-to-image | [Qwen-Image](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-qwen-image/README.md) | H100 | [Create Endpoint](https://console.nebius.com/serverless/endpoint/create?template=qwen-image) |
| Text-to-image | [Z-Image-Turbo](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-z-image-turbo/README.md) | H100 | [Create Endpoint](https://console.nebius.com/serverless/endpoint/create?template=z-image-turbo) |
| Image editing | [Qwen-Image-Edit-2511](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-qwen-image-edit-2511/README.md) | H100 | [Create Endpoint](https://console.nebius.com/serverless/endpoint/create?template=qwen-image-edit-2511) |
| Text-to-video | [Wan2.1-T2V-1.3B](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-wan21-t2v-1-3b/README.md) | H100 | [Create Endpoint](https://console.nebius.com/serverless/endpoint/create?template=wan21-t2v-1-3b) |
| Image-to-video | [Wan2.2-I2V-A14B](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-wan22-i2v-a14b/README.md) | H100 | [Create Endpoint](https://console.nebius.com/serverless/endpoint/create?template=wan22-i2v-a14b) |
| Text-to-speech | [Kokoro-82M](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-kokoro-82m/README.md) | L40S | [Create Endpoint](https://console.nebius.com/serverless/endpoint/create?template=kokoro-82m) |
| Text-to-speech | [Chatterbox](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-chatterbox/README.md) | H100 | [Create Endpoint](https://console.nebius.com/serverless/endpoint/create?template=chatterbox) |
| Text-to-music | [ACE-Step 1.5](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-ace-step-1-5/README.md) | H100 | [Create Endpoint](https://console.nebius.com/serverless/endpoint/create?template=ace-step-1-5) |

## Homework 1: GPU Media Worker

Generate one media asset with any model from the [table above](#models-used-in-this-module), measure it, and compare its cost with a per-call API.

* Notebook or `.py` script, your choice
* Report your model, generation time and cost per asset, and the creative idea behind your asset
* Find the per-call API price for the same model and compute your self-host breakeven in number of assets

Questions and submission: [homework.md](../cohorts/2026/homework/01-consistent-media-generation-on-serverless-gpus/homework.md).

## Capstone Proposal

Write a one-page proposal for your capstone: problem, records, two GPU media models and a budget estimate. Template: [proposal-template.md](../projects/proposal-template.md) in [projects](../projects).

## Module structure

One page per lesson at the module root, all code flat in `code/` (files import each other), guide images in `images/`.

```text
01-consistent-media-generation-on-serverless-gpus/
├── README.md                       # lessons index, models, setup
├── 01-landscape.md
├── 02-first-generation.md
├── 03-containers.md
├── 04-serving-vs-batch.md
├── 05-measuring-costs.md
├── images/                         # guide images: NN-slug-MM-what.png
└── code/
    ├── .env                        # you create it: Endpoint URL + token (notebooks write it; git-ignored)
    ├── first_image_local.ipynb     # 02: your laptop → Endpoint
    ├── first_image_devlab.ipynb    # 02: JupyterLab on a GPU
    ├── image.py                    # 02: ImageGenerator client (generate_image), comparison strip, request_id
    ├── batch_inference.py          # 04, 05: n images through an Endpoint with one config → assets, metrics.json, trace.jsonl; CLI --n 3
    ├── common.py                   # 02–05: setup_endpoint(), request(), wait_ready(), .env helpers, config checks, metrics/trace writers, calculate_cost()
    └── docker/                     # 03, 04: Dockerfiles
```

## Setup

Do this once, before any lesson in this module. Every lesson needs steps 1–3, whether you call an Endpoint from your laptop or work in JupyterLab on a GPU.

1. **Join the [Nebius AI Builder Program](https://dev.nebius.com/builders)** and get your promo code. It's free.
2. **Create a Nebius AI Cloud account and add payment details** at [console.nebius.com](https://console.nebius.com/). When you add a bank card, Nebius charges $25 and adds this amount to your account balance. You need payment details before you can apply a promo code. [Sign up](https://docs.nebius.com/signup-billing/sign-up)
3. **Apply your Builder Program promo code:** in the console, go to [Billing](https://console.nebius.com/billing/), click **Apply promo code**, enter the code and click **Apply promo code**. Your balance increases by the code's amount; check it in the [transactions list](https://console.nebius.com/billing/transactions). If you don't see the button, check that you added payment details. Apply the code before its expiration date. [How to use a promo code](https://docs.nebius.com/signup-billing/payments/promo-codes#how-to-use-a-promo-code)
4. **Clone this repository** and install [uv](https://docs.astral.sh/uv/getting-started/installation/) (`uv` installs Python 3.12 for you).
5. **Start Jupyter** from the repository root:

   ```bash
   uv sync --extra notebooks
   uv run jupyter lab
   ```

Endpoint URL and token: the notebooks ask for them and save them to `code/.env`, which only you can read and Git ignores. Never commit tokens or paste them into chats or screenshots.

Stop or delete every Endpoint, Job and JupyterLab when you finish: an Endpoint bills while it's `RUNNING`, even when it serves no requests.

# Community Notes

Did you take notes? You can share them here:

* Add your notes here!
