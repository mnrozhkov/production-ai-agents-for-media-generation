---
video_url: ""
prev_url: 01-agent-loop.md
next_url: 03-gpu-tools.md
---
# 2.2 Tools and Structured Outputs

An agent that makes ad keyframes will, sooner or later, call its image tool with a size the model doesn't support, wait less than the image takes, or ask for the same image twice. Each of those can cost GPU time for nothing. In this lesson you design the image tool so none of them does.

## What you will learn

* Read the JSON schema a model gets from a tool's signature and docstring, and write the docstring for the model
* Tell three kinds of retry apart: validation, transport and uncertain submission, and what each one costs
* Set a tool timeout that doesn't pay for the same image twice
* Reuse a finished generation with a spec hash instead of paying for it again
* Return a reference to an asset, not its bytes
* Get a typed `Plan` from an LLM step with `output_type`

## Outcome

You're done when:

* the model fixes an invalid call after one validation retry,
* a second identical `generate_image` call is a cache hit, with no GPU time and no cost.

Time: about 25 minutes.
Cost: free in `"fake"` mode. A few cents of tokens in `"fake-gpu"` mode [measure]. No GPU in this lesson.

## Before you start

* Complete the **Setup** section of the [module README](README.md) once.
* For `"fake-gpu"` mode: a Token Factory API key in `code/.env`. The notebook asks for it if it's missing.

## Run the lesson

Open [`code/nb-tools.ipynb`](code/nb-tools.ipynb) in Jupyter and run it top to bottom. Every explanation, exercise and expected output is in the notebook.

The mode cell at the top chooses the LLM:

| `MODE` | LLM | Image backend | Cost |
| --- | --- | --- | --- |
| `"fake"` | A scripted model: a fixed list of replies | Placeholder PNG | Free, no network |
| `"fake-gpu"` | A real LLM on Token Factory, for the validation and plan demos | Placeholder PNG | A few cents of tokens |

`"fake-gpu"` means the GPU is fake, not the LLM. To switch, edit the mode cell and run it again. No kernel restart needed.

The notebook has one section per demo:

1. **The tool signature is the schema:** print what the model receives
2. **Three kinds of retry:** an invalid size, fixed by the model in one retry
3. **Tool timeouts:** a timeout below the generation time sends two GPU requests
4. **Cache keys:** a second identical call is a cache hit; one changed byte in an input is a miss
5. **Return references, not bytes:** what the model sees vs what the run folder holds
6. **Structured outputs:** a typed `Plan` from the brief

## Where the results are

Every demo writes its own folder `code/runs/02-multi-agent-media-workflows/<run_id>/` with:

* `events.jsonl`: model requests and tool calls, with `cache_hit`, `t_gen_s` and `cost_usd`
* `keyframe-<first 12 characters of the spec hash>.png`: the placeholder keyframes

With `BUCKET` set in `code/.env`, the keyframes are also copied to `s3://<bucket>/runs/02-multi-agent-media-workflows/<run_id>/`.

## Files used

| File | What it does here |
| --- | --- |
| [`code/nb-tools.ipynb`](code/nb-tools.ipynb) | The lesson |
| [`code/tools.py`](code/tools.py) | `generate_image`, `spec_hash()`, `AssetRef`, `ToolDeps`, `Store`, `token_factory_model()` |
| [`code/models.py`](code/models.py) | `Brief`, `Plan` and the pinned model specs |
| [`code/fake.py`](code/fake.py) | `scripted_model()`, `call()`, `PlaceholderImages`, `block_real_models()` |
| [`code/tracing.py`](code/tracing.py) | `Tracer`, `run_traced()`, `last_tool_return()` |
| [`assets/brief.yaml`](assets/brief.yaml) | The ad brief |

## Next steps

* Add one argument to `generate_image` in `tools.py`, with a type constraint and a docstring line. Run the schema cell and find it in what the model receives.
* Homework 2 needs a cache hit on a second identical GPU call, with the keyframe's content hash in the key. `spec_hash()` is the starting point.
* Next lesson: [2.3 GPU Workloads as Agentic Tools](03-gpu-tools.md).

<!-- nav:start -->
[← Previous: 2.1 The Agent Loop: Deep Dive](01-agent-loop.md) · [Next: 2.3 GPU Workloads as Agentic Tools →](03-gpu-tools.md)
<!-- nav:end -->
