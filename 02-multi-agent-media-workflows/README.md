<p align="center">
  <img src="../images/module-2-cover.png" alt="Module 2, Make it intelligent: Multi-Agent Media Workflows. The agent loop, GPU models as typed tools over MCP, long-running tools, planner, generator and reviewer. You ship Homework 2: Media Agent" width="100%">
</p>

# Module 2: Multi-Agent Media Workflows

Welcome to Module 2 of Production AI Agents for Generative Media! This week we make it intelligent: give an agent GPU workloads as tools, keep the agent on CPU, and build a planner, generator and reviewer loop that regenerates an asset instead of accepting the first output.

> [!NOTE]
> This module builds on Module 1: the Endpoints, Jobs, Object Storage bucket and cost terms from [1.4](../01-consistent-media-generation-on-serverless-gpus/04-serving-vs-batch.md) and [1.5](../01-consistent-media-generation-on-serverless-gpus/05-measuring-costs.md). The agent's LLM runs on [Nebius Token Factory](https://docs.tokenfactory.nebius.com/quickstart), an OpenAI-compatible API.

> [!WARNING]
> Make sure you stop or delete your resources (Endpoints, Jobs) when you finish the module to avoid charges. An Endpoint bills while it's RUNNING, even when the agent isn't calling it. Video is the most expensive asset: keep the fast path to two videos or fewer.

* Video Link (Coming Soon)
* Slides (Coming Soon)

<p align="center">
  <img src="../images/diagram-agent-gpu-tools.png" alt="A CPU agent with planner, generator and reviewer calls GPU models as typed tools: image, voice, video and speech to text. Those tools run on Serverless AI Endpoints and Jobs, and results go to object storage" width="100%">
</p>

## What you will learn

By the end of this module, you will be able to:

* Explain the loop an agent framework runs for you (model request → tool calls → tool results → output) and step through it with PydanticAI
* Choose between agent, workflow and hybrid architectures for a media task
* Design tools with typed inputs and outputs, validation, retries, timeouts and idempotency
* Wrap GPU workloads as agent tools: inference on an Endpoint, batch inference or fine-tuning as a Job, called directly or over MCP
* Handle long-running tools: submit, poll, resume, without paying for the same GPU work twice
* Write a versioned rubric from the brief and build a planner, generator and reviewer loop with attempt caps, runtime gates and metrics

Default path for the whole module: one [PydanticAI](https://pydantic.dev/docs/ai/) agent on Token Factory, the [Sana](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-sana/README.md) Endpoint from Module 1 for images, [Wan2.2-I2V-A14B](https://github.com/nebius/serverless-ai-cookbook/blob/main/templates/endpoint-wan22-i2v-a14b/README.md) for video. Every model request, tool call and check writes one event to `events.jsonl` in the run folder. Every lesson script also runs with `--fake`: fully offline, with a scripted model and placeholder assets, so you can read the code path without credits.

## Lessons

* [2.1 The Agent Loop: Deep Dive](01-agent-loop.md)
* [2.2 Tools and Structured Outputs](02-tools.md)
* [2.3 GPU Workloads as Agentic Tools](03-gpu-tools.md)
* [2.4 Long-running Tools: Submit, Poll, Resume](04-long-running-tools.md)
* [2.5 Planner, Generator, Reviewer: the Judge Loop](05-judge-loop.md)

## Homework 2: Media Agent

Build a media agent for a short coffee ad video: plan → keyframe → judge → regenerate → video, with a full trace of every step.

* The agent judges the keyframe and regenerates on `RETRY` instead of accepting the first output. Show recovery on a fixture keyframe; your live run may pass on the first attempt.
* Video runs as a long-running tool with a ticket
* Tools called directly or over MCP, your choice
* Report the cost per video, split by step, and whether you chose an agent, a workflow or a hybrid for this brief, and why

Questions and submission: [homework.md](../cohorts/2026/homework/02-multi-agent-media-workflows/homework.md).

## Module structure

One page per lesson at the module root, all code flat in `code/` (files import each other), task data in `assets/`.

```text
02-multi-agent-media-workflows/
├── README.md                       # lessons index, setup
├── 01-agent-loop.md
├── 02-tools.md
├── 03-gpu-tools.md
├── 04-long-running-tools.md
├── 05-judge-loop.md
├── assets/
│   ├── brief.yaml                  # the lesson brief: the Brief fields Homework 2 uses too
│   ├── rubric.yaml                 # 2.5: rubric v1, criteria that read Brief fields
│   ├── fixtures/                   # 2.5: bad and good keyframe with their expected verdicts
│   └── worked-trace/               # a hand-written trace and its expected totals
└── code/
    ├── .env                        # you create it: Token Factory key, Endpoint URLs and tokens (git-ignored)
    ├── nb-agent-loop.ipynb         # 2.1: be the LLM, swap in Token Factory, agent.iter(), limits, state
    ├── nb-tools.ipynb              # 2.2: schema, validation retry, timeout, cache hit, typed plan
    ├── nb-gpu-tools.ipynb          # 2.3: the Sana Endpoint as generate_image; MCP also try
    ├── mcp_server.py               # 2.3: the same image tool over MCP (stdio)
    ├── video_tool.py               # 2.4: deferred video tool, ticket, reconcile, resume
    ├── nb-judge.ipynb              # 2.5: rubric v1 and the judge on both fixtures
    ├── review_loop.py              # 2.5: planner, generator, reviewer loop with caps and budget guard
    ├── tools.py                    # 2.2–2.5: generate_image, spec hash, Sana backend, settings, Store
    ├── tracing.py                  # all: events.jsonl writer, run_traced(), totals, contract check, metrics.json
    ├── fake.py                     # all: scripted model, placeholder image and video backends
    └── models.py                   # all: pinned model specs (LLM, judge, Sana, Wan2.2), Brief, Plan
```

Every run writes to `code/runs/<lesson>/<run_id>/`, for example `code/runs/02-tools/<run_id>/`: `events.jsonl`, `messages.json`, `metrics.json`, `tickets/` and the assets. With `BUCKET` set in `code/.env`, assets and run files are also copied to `s3://<bucket>/runs/<lesson>/<run_id>/`.

## Setup

Do this once, before any lesson in this module.

1. **Module 1 setup:** a Nebius account with credits and the repository cloned. See the [Module 1 README](../01-consistent-media-generation-on-serverless-gpus/README.md).
2. **An LLM API key.** The default is [Nebius Token Factory](https://docs.tokenfactory.nebius.com/quickstart); the notebooks ask for the key once and save it to `code/.env` as `NEBIUS_API_KEY`. Any OpenAI-compatible API with tool calling works too: set `LLM` in `code/models.py` (see [2.1](01-agent-loop.md#use-another-llm)).
3. **Install and start Jupyter** from the repository root:

   ```bash
   uv sync --extra notebooks
   uv run jupyter lab
   ```

4. **Scripts** run from the code folder:

   ```bash
   cd 02-multi-agent-media-workflows/code
   uv run python review_loop.py --fake
   ```

Every notebook has a `MODE` cell and every script a `--fake` flag: a scripted model and placeholder images and videos, fully offline. Start there, then switch to the real models.

Endpoints: lesson 2.3 uses the [Sana](https://console.nebius.com/serverless/endpoint/create?template=sana) template, 2.4 the [Wan2.2-I2V-A14B](https://console.nebius.com/serverless/endpoint/create?template=wan22-i2v-a14b) template, 2.5 both. Start each one when its lesson needs it, and stop it when the lesson ends. Check with `nebius ai endpoint list` that nothing is still `RUNNING`.

# Community Notes

Did you take notes? You can share them here:

* Add your notes here!
