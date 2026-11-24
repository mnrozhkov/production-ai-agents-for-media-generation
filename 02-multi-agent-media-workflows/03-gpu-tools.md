---
video_url: ""
prev_url: 02-tools.md
next_url: 04-long-running-tools.md
---
# 2.3 GPU Workloads as Agentic Tools

An ad agent spends most of its time waiting: for the LLM to answer, for an image to render. If the agent itself ran on a GPU machine, you'd pay GPU rates for all of that waiting. In this lesson we keep the agent on your laptop's CPU and give it the Sana image model on a Nebius Serverless Endpoint as a tool, with the Endpoint's token kept away from the model. MCP is an optional extension at the end. You're done when your agent calls a GPU tool and each trace event records the workload type, its ID and the generation time, and the generated image opens from its URI.

Before you start: complete the **Setup** section of the [module README](../02-multi-agent-media-workflows/README.md) once.

Code: [`code/nb-gpu-tools.ipynb`](code/nb-gpu-tools.ipynb) for the main path, [`code/mcp_server.py`](code/mcp_server.py) for the MCP extension. The GPU tool lives in the shared [`code/tools.py`](code/tools.py).

Cost: one Sana Endpoint on a preemptible L40S, billed per second while it's `RUNNING`, plus a few images. Our run: Endpoint startup [measure], readiness wait [measure], [measure] per image, $[measure] in total [calc: metrics.json × rate on the run day]. Prices change; check [Compute pricing](https://docs.nebius.com/compute/resources/pricing). The notebook ends with the teardown.

## Control plane vs compute and data plane

<p align="center">
  <img src="../images/diagram-agent-gpu-tools.png" alt="A CPU agent calls GPU models as typed tools on Serverless AI Endpoints and Jobs; results go to object storage" width="100%">
</p>

- **Control plane, on CPU:** the agent loop and its LLM calls to Token Factory. Cheap to run, mostly waiting.
- **Compute plane, on GPU:** the Endpoint that renders images. It only works when a tool calls it.
- **Data plane:** the run folder and your bucket. Tools write assets there and return a reference (2.2).

## Kinds of GPU tool

"A GPU tool" can mean seconds or hours:

| Kind | Serverless resource | Typical duration | Call pattern | In this course |
| --- | --- | --- | --- | --- |
| Inference: one image | Endpoint | seconds [measure] | Synchronous: call, wait, result | This lesson |
| Inference: one video | Endpoint | minutes [measure] | Submit, poll, resume | 2.4 |
| Batch inference | Job | minutes to hours | Submit, poll | Module 3 |
| Fine-tuning | Job | hours | Submit, poll | 4.2 (optional) |

This lesson builds the first row. The other rows use the same idea, with a long-running call pattern instead of a wait.

## How the agent calls a tool: function or MCP

The same tool can reach the agent two ways:

- **A Python function** in the agent's process. The simplest option when one agent uses the tool. Default path.
- **An MCP server**: a separate process that serves tools over the [Model Context Protocol](https://modelcontextprotocol.io/). Pick it when other clients, another agent or your IDE, should use the same tools.

The tool code doesn't change between the two: `mcp_server.py` calls the same `image_asset()` function as the PydanticAI tool.

## Credentials

The Endpoint token never goes into a prompt, a tool argument or a trace event. The notebook asks for the URL, token and Endpoint ID once and saves them to `code/.env` (mode 600, git-ignored). The tool side reads them:

```python
    env = settings("SANA_BASE_URL", "SANA_TOKEN", "SANA_ENDPOINT_ID", secret=("SANA_TOKEN",))
    images = SanaEndpoint(env["SANA_BASE_URL"], env["SANA_TOKEN"], env["SANA_ENDPOINT_ID"])
```

`ToolDeps` carries the backend into the tool through `ctx.deps`, which the model never sees. After the run, the notebook checks the trace file for the token:

```python
    assert env["SANA_TOKEN"] not in tracer.path.read_text(), "token leaked into the trace"
```

## Tool lifecycle

Can the agent start a model just long enough to make one asset? A Job does that: it starts, runs and stops by itself. An Endpoint the agent starts bills until something stops it, and an agent that crashes won't. So in this course you start and stop Endpoints yourself, outside the agent, and the tool only checks that the Endpoint is ready before its first call:

```python
    def wait_ready(self, deadline_s: float = 900) -> float:
        """Poll /v1/models until the server lists a model; return the seconds waited.

        RUNNING in the console means the container started, not that the weights are loaded.
        """
```

## Default path: Sana as generate_image

Start the Sana Endpoint from its [template](https://console.nebius.com/serverless/endpoint/create?template=sana) and wait for `RUNNING`. Set `MODE = "live"` and `GPU_PRICE_PER_HOUR` to the hourly price of the Endpoint's GPU and preset, then run the notebook.

The agent and the tool are the same as in 2.2. Only the backend behind `generate_image` changes, from a placeholder to the Endpoint:

```python
agent = Agent(
    model,
    deps_type=ToolDeps,
    tools=[generate_image],
    instructions="You make one keyframe image for a video ad brief. Call generate_image once.",
)
result = await run_traced(agent, f"Brief: {brief.model_dump_json()}", tracer=tracer, deps=deps)
```

The trace now says where the work ran and how long it took:

```text
model_request  llm       [LLM model ID]                   t_gen_s=None cost=[measure]
tool_call      endpoint  [your Endpoint ID]               t_gen_s=[measure] cost=[measure]
model_request  llm       [LLM model ID]                   t_gen_s=None cost=[measure]
```

- `workload`: `endpoint`, `job`, `llm` or `none` (the placeholder).
- `workload_id`: which Endpoint, Job or LLM did the work. When the bill looks wrong, this is where you start.
- `t_gen_s`: the request latency for one image, the same `t_gen` as in Module 1. Its cost is `rate × t_gen` [calc], charged once, on this event.

The image opens from the URI in `asset_ref`: a local path, or `s3://…` when `BUCKET` is set in `code/.env`.

Optional: one cell generates 10 more images and prints the median and p95 generation time. 2.2 uses the p95 for the tool timeout, 2.5 for the budget estimate. Our numbers: median [measure], p95 [measure].

## When it fails

| Symptom | Cause | Fix |
| --- | --- | --- |
| `wait_ready` runs for minutes, then `TimeoutError` | The Endpoint is still loading weights, or failed to start | Check its logs in the console. Stop it if it failed |
| `401` or `403` | Wrong or missing token | Fix `SANA_TOKEN` in `code/.env` |
| `502`/`503` for a while after `RUNNING` | Weights still loading | Call `wait_ready()` first. During a generation, `generate()` treats `502`/`503`/`504` as uncertain and raises `UncertainSubmission` instead of resending |
| `Timed out after … seconds` in the trace, two GPU requests | Tool timeout below the generation time | Raise it above p95 (2.2) |
| A size the model doesn't accept | The model asked for a size outside `Size` | A validation retry fixes it before any GPU call (2.2) |
| The bill keeps growing after the lesson | The Endpoint is still `RUNNING` | Teardown below |

[measure: trigger each once and paste the error text learners will see]

## Teardown

Stop the Endpoint now: in the console, open it and click **Stop**. Then check that nothing is still running:

```bash
nebius ai endpoint list
```

## Also try: tools behind an MCP server

`mcp_server.py` serves the same image tool over stdio:

```python
@server.tool()
async def generate_image(prompt: Prompt, seed: int = SANA.seed, size: Size = "832x480") -> dict:
```

It uses the MCP Python SDK 2.x, where `FastMCP` is now `MCPServer` (`from mcp.server.mcpserver import MCPServer`). The token stays in the server process: it reads `code/.env` itself and never asks for input, because over stdio its input is the protocol.

From the notebook, PydanticAI starts the server as a subprocess and uses it as a toolset. MCP doesn't pass the framework's tool call ID to the server, so a small hook on the client side writes the trace event, where the ID is known:

```python
async def trace_mcp_call(ctx, call_tool, name, tool_args):
    """Call the MCP tool, then write the tool_call event with the framework's call ID."""
    result = await call_tool(name, tool_args, None)
    asset = AssetRef.model_validate(result if isinstance(result, dict) else json.loads(result))
    trace_asset(deps, ctx.tool_call_id, name, asset)
    return result
```

From an IDE agent, register the server with its own config. Most clients take a JSON entry like this (EXAMPLE, NOT RUN; check your client's docs for the file name):

```json
{
  "mcpServers": {
    "media-tools": {
      "command": "uv",
      "args": ["run", "python", "mcp_server.py"],
      "cwd": "<repo>/02-multi-agent-media-workflows/code"
    }
  }
}
```

Then ask the IDE agent for a keyframe and look for the PNG in `code/runs/03-gpu-tools/mcp/`.

## Notes

- PydanticAI's tool timeout isn't enforced for MCP tools; set deadlines on the server or transport ([tool timeout](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#tool-timeout)).
- `MCPServerStdio(..., process_tool_call=...)` on the pinned PydanticAI version: [verify].
- Remote MCP servers with authentication: 4.4.

Next: [2.4 Long-running Tools: Submit, Poll, Resume](04-long-running-tools.md), where the tool is a video that takes minutes.

<!-- nav:start -->
[← Previous: 2.2 Tools and Structured Outputs](02-tools.md) · [Next: 2.4 Long-running Tools: Submit, Poll, Resume →](04-long-running-tools.md)
<!-- nav:end -->
