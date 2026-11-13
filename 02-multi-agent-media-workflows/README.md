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

## Module structure (proposed)

One page per lesson at the module root, all code flat in `code/`, images in `images/`.

```text
02-multi-agent-media-workflows/
├── README.md                       # lessons index
├── 01-agent-loop.md
├── 02-tools.md
├── 03-gpu-tools.md
├── 04-long-running-tools.md
├── 05-judge-loop.md
├── images/
└── code/
    ├── README.md                   # how to run each file, env vars, the two fake modes
    ├── models.yaml                 # every model ID and revision for Module 2, pinned
    ├── trace.py                    # emit() → events.jsonl, per the trace contract
    ├── check_trace.py              # validates events.jsonl: fields, no cost on node events, steps, totals
    ├── fake.py                     # scripted model (FunctionModel) + placeholder GPU tools
    ├── agent_loop.py               # 2.1
    ├── bench_tool_calling.py       # 2.1
    ├── tools.py                    # 2.2: typed tools, spec hash, bucket cache, retry classes
    ├── gpu_tools.py                # 2.3: Endpoint tool, Job tool
    ├── mcp_server.py               # 2.3 also try
    ├── tickets.py                  # 2.4: ticket states, reconcile
    ├── video_tool.py               # 2.4: async Videos API; Job variant
    ├── kill_resume.sh              # 2.4: one run per restart boundary
    ├── rubric.yaml                 # 2.5: rubric v1, criteria from brief fields with pass and fail examples, version
    ├── judge.py                    # 2.5: typed verdict with rubric_version, judge_model, reasons
    ├── review_loop.py              # 2.5: one agent, role prompts
    ├── multi_agent.py              # 2.5 also try: planner + reviewer agents
    └── fixtures/
        ├── bad_keyframe.png        # 2.5: breaks the brief on purpose
        └── bad_keyframe.json       # expected verdict and why
```

# Community Notes

Did you take notes? You can share them here:

* Add your notes here!
