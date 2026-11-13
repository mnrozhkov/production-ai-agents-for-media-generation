## Homework 2: Media Agent

> [!NOTE]
> You need the Module 1 setup: a Nebius project with credits and an Object Storage bucket. See [Module 2](../../../../02-multi-agent-media-workflows/README.md).
> Stop or delete every Endpoint and Job when you're done. A forgotten Endpoint bills while it's RUNNING.

In this homework, you build an agent that turns a written brief into a short video ad, checks its own keyframe, regenerates it when it isn't good enough, and leaves a trace of every decision.

### The brief

Save this as `brief.yaml` next to your code. You can change the copy, not the structure.

```yaml
product: Byte Brew, a fictional coffee brand
audience: software developers
format: 16:9
shot: a developer at a desk late at night takes the first sip from a Byte Brew cup and relaxes
must_show: the cup, the developer, a laptop
must_not: real brand names, logos, readable text or code
```

### Setup

* One [PydanticAI](https://pydantic.dev/docs/ai/) agent. The LLM and the judge run on [Nebius Token Factory](https://docs.tokenfactory.nebius.com/quickstart). The judge needs a vision model.
* GPU tools from Module 1, one per role:
  * Keyframe (text-to-image): Sana by default. Also fine: Qwen-Image, Z-Image-Turbo.
  * Video (image-to-video): Wan2.2-I2V-A14B.
* Call the tools as plain functions or over MCP, your choice.
* Rubric: start from the 2.5 rubric v1 (`rubric.yaml`) or write your own. Give it a version.
* Use a `.py` script or a notebook.

### Build the agent

The agent runs these steps:

1. **Plan:** turn `brief.yaml` into an image prompt and a motion prompt
2. **Keyframe:** generate one image
3. **Judge:** score it against the brief. Typed verdict with a score, reasons and a fix:
    * `PASS`: go to video
    * `RETRY`: generate again with the fix, up to your attempt cap
    * `WEAK`: stop and keep the keyframe for diagnosis. It never goes to video.
    * A reply that doesn't parse: ask the judge again. That's not a new generation, and it never counts as `PASS`.
4. **Video:** image-to-video from the `PASS` keyframe, as a long-running tool (2.4):
    * Submit with `POST /v1/videos`, poll `GET /v1/videos/{video_id}`, download the content, copy it to your bucket, then delete it from the Endpoint
    * The ticket holds the Endpoint ID and the video ID. Write it with state `intent` before you submit, and save the video ID right after.
5. **Check:** the video file exists, and its duration and resolution match what you requested

### Show recovery with a fixture

Make one keyframe that breaks the brief on purpose: for example, generate it without the cup, or with readable text on the screen. Save it as `fixture_keyframe.png` and write down the verdict you expect (`RETRY`).

Start one run with the fixture as attempt 1. The judge must return `RETRY`, and the loop must regenerate until it gets `PASS`. Your normal run may pass on the first attempt; that's fine.

### Requirements

* Recovery from the fixture, visible in the trace
* At most 2 video generations in the whole homework, including retries. Video is the most expensive asset.
* A second identical call to any GPU tool is a cache hit, not a new generation (2.2). The cache key includes the content hash of the keyframe you pass to the video model.
* Every model request, tool call and check writes one event to `events.jsonl` with: `event_id`, `kind`, `name`, `call_id`, `workload` and `workload_id`, `attempt`, `verdict`, `t_gen_s`, `cost_usd`. Judge events also carry `rubric_version`, `judge_model` and `reasons`.
* Watch the final video once yourself, and write one sentence on whether it matches the brief. The judge only saw the keyframe.

### Compute the cost per video

Use the Module 1 terms. See [1.5](../../../../01-consistent-media-generation-on-serverless-gpus/05-measuring-costs.md).

* For every GPU generation, including rejected keyframes: `c = rate × t_gen`. For the video, `t_gen` runs from submit to `completed`, so it includes queue time.
* For every LLM and judge call: tokens × the Token Factory price per token
* Charge each execution once, on the event that paid for it. Cache hits, status polls and checks cost $0.
* Marginal cost per video: the sum of `cost_usd` over the run, split by step. The fixture run doesn't count.
* Rejected share: `t_gen` of keyframes that didn't get `PASS` ÷ `t_gen` of all GPU generations in the run

The marginal cost assumes running, ready Endpoints. Startup, readiness wait and idle time are fixed costs; report them separately if you measured them, and don't add them to the marginal cost. Rates: [Nebius pricing](https://docs.nebius.com/compute/resources/pricing).

### Optional

* Kill the agent while the video is pending, resume it with the run ID, and check the trace for a second video generation. Then try a kill right after submit, before the video ID is saved, and see what your reconcile step does (2.4).
* Run the same brief with one agent and with a planner plus a separate reviewer agent, and compare steps and cost (2.5)

### Question 1

Did you build an agent, a workflow or a hybrid for this brief? Why? Two or three sentences.

### Question 2

Which image model did you use for the keyframe?

- Sana
- Qwen-Image
- Z-Image-Turbo

### Question 3

How did your agent call the GPU tools?

- Plain Python functions
- Over MCP
- Both

### Question 4

What did the judge return on your fixture keyframe?

- `RETRY`
- `WEAK`
- `PASS`
- The reply didn't parse

### Question 5

How many keyframe attempts did your normal run need before the judge returned `PASS`?

### Question 6

How many steps did your normal run take, counting model requests and tool calls?

### Question 7

What was your marginal cost per video, in USD?

### Question 8

Which step cost the most?

- Plan (LLM)
- Keyframe image
- Judge
- Video

### Question 9

What was your rejected share, in percent?

### Question 10

Link to your repository with the code, `brief.yaml`, `fixture_keyframe.png`, `events.jsonl` and the final video.

## Learning in Public

Share your video and one thing your agent's trace taught you in a short post. Use `#[placeholder]`, tag DataTalksClub and Nebius, and submit the post link with your homework. Don't share credentials, Endpoint tokens or bucket keys.

See the [learning in public guide](https://datatalks.club/docs/courses/zoomcamp-logistics/learning-in-public/) for more ideas.

### Example post for LinkedIn

```text
Completed Module 2 of Production AI Agents for Generative Media!

I built an agent that turns a brief into a short video ad (attached).
It planned, generated, judged and regenerated: <N> keyframe attempts before PASS,
<M> steps in total, $<cost> per video.

What surprised me: <one thing from the trace>.

My code: <LINK>

#[placeholder] @DataTalksClub @Nebius
```

## Submit the results

* Submit your results here: [placeholder]
* All questions are about your own run. Any honest answer counts.
