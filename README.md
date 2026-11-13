<p align="center">
  <img src="images/course-cover.png" alt="Production AI Agents for Generative Media: build, scale and monitor agents that run image, video and audio models on serverless GPUs. Free, open source (MIT), project-based. Nebius × DataTalks.Club" width="100%">
</p>

<p align="center">
<b>Starts Mon 16 Nov 2026 · <a href="https://courses.datatalks.club/register/production-ai-agents-generative-media/">Apply here</a></b>
</p>

> How do you build a production AI system when an agent can launch expensive, slow, failure-prone GPU workloads that produce large media artifacts?

Media agents are an engineering problem, not a prompting problem. In this course, you build one production system that handles it: a **Personalized Media Factory**.

1. **Input:** a brief, made of a task prompt, a data record and constraints (format, style, budget)
2. **Agent (LLM):** plans the asset and writes its text, such as a card message or an ad script
3. **GPU models:** generate an image plus voice or video
4. **Check:** regenerates the asset when it is not good enough
5. **Scale:** from 1 to 1,000+ records, with a cost per asset

<p align="center">
  <img src="images/diagram-media-factory.png" alt="The Personalized Media Factory: a brief goes to an agent (Module 2), GPU models generate image, voice or video (Module 1), a check (Module 4) sends assets that aren't good enough back to the agent to revise and regenerate, and the factory scales from 1 to 1,000+ records with a cost per asset (Module 3)" width="100%">
</p>

## What You'll Learn

* How to run image, video and audio models as Jobs and Endpoints on serverless GPUs, without managing a cluster
* How to give agents GPU models as tools (MCP) and build generate → evaluate → regenerate loops
* How to fan out batch workloads on spot GPUs, handle failures and report cost per asset
* How to add evals, tracing and SLOs so you can say what a media agent costs and when it is good enough to ship

## Dates

All times are CET. Modules run Monday to Sunday. More on this cohort: [cohorts/2026](cohorts/2026).

| Date | What | Due / live session |
|---|---|---|
| Tue 3 Nov 2026 | Workshop: "Build a Media Agent: CPU Agent, GPU Models as Tools" | Live, 16:30-18:00 · [Register](https://luma.com/tj2nbrl2) · [Code](https://github.com/DataTalksClub/build-media-agent-workshop) |
| Tue 10 Nov 2026 | Pre-Course Live Q&A | Live, 17:00-18:00 · [Register](https://luma.com/i4x6x2zy) |
| Mon 16 Nov 2026 | Course launch | Live, 17:00-18:00 · [Register](https://luma.com/xey7h50c) |
| Mon 16 - Sun 29 Nov 2026 | [**Module 1: Consistent Media Generation on Serverless GPUs**](01-consistent-media-generation-on-serverless-gpus/README.md) (two weeks) | Homework 1 and capstone proposal |
| Mon 30 Nov - Sun 6 Dec 2026 | [**Module 2: Multi-Agent Media Workflows**](02-multi-agent-media-workflows/README.md) | Homework 2, due Sun 6 Dec |
| Mon 7 - Sun 13 Dec 2026 | [**Module 3: Batch Media Generation at Scale**](03-batch-media-generation-at-scale/README.md) | Homework 3 (leaderboard), due Sun 13 Dec |
| Mon 14 - Sun 20 Dec 2026 | [**Module 4: Evaluation, Observability and Production**](04-evaluation-observability-production/README.md) | Capstone brief |
| Fri 15 Jan 2027 | Capstone submission, Attempt 1 | Capstone |
| Fri 22 Jan 2027 | Peer review, Attempt 1 | Capstone |
| Fri 29 Jan 2027 | Capstone submission, Attempt 2 | Capstone |
| Fri 5 Feb 2027 | Final peer review, Attempt 2 | Capstone |

Materials and dates may change.

## Modules

<p align="center">
  <img src="images/diagram-course-roadmap.png" alt="Course structure: Module 0 Welcome, Module 1 Make it work, Module 2 Make it intelligent, Module 3 Make it scale, Module 4 Make it production-ready, then the capstone" width="100%">
</p>

### Module 0: Welcome

Tue 10 Nov 2026, live. Why media agents are an engineering problem, not a prompting problem.

* The factory end to end
* Real projects from Nebius customers and partners
* **You leave with:** your starting point and setup

### [Module 1: Consistent Media Generation on Serverless GPUs](/01-consistent-media-generation-on-serverless-gpus)

Mon 16 - Sun 29 Nov 2026 (two weeks)

* Media model landscape
* First media generation in a few minutes
* Containers for generative AI
* Serving vs batch inference
* Measuring startup, load and generation costs
* **Homework 1: GPU Media Worker** (your creative idea in, an asset plus cost per asset and self-host breakeven out)
* **Capstone proposal**

### [Module 2: Multi-Agent Media Workflows](/02-multi-agent-media-workflows)

Mon 30 Nov - Sun 6 Dec 2026

* The agent loop from scratch
* GPU models as typed tools over MCP
* Long-running tools
* Planner / generator / reviewer
* **Homework 2: Media Agent** (a 10-second ad plus an execution trace), due Sun 6 Dec

### [Module 3: Batch Media Generation at Scale](/03-batch-media-generation-at-scale)

Mon 7 - Sun 13 Dec 2026

* Fan-out with Jobs
* Spot GPUs and design for interruption
* Optimizing cost per asset: cold start, VRAM and speed
* Breaking the pipeline on purpose
* **Homework 3: Factory Scale Challenge** (200 records over 10 parallel Jobs, public leaderboard), due Sun 13 Dec

### [Module 4: Evaluation, Observability and Production](/04-evaluation-observability-production)

Mon 14 - Sun 20 Dec 2026

* Evaluating generative systems: system, agent and media
* Judge calibration and regression gates
* Observability and tracing
* Designing for production: architecture, SLO, dashboard and alerts
* Production concerns and the capstone brief: auth, quotas, abuse, prompt injection, content safety, licensing and provenance
* No homework: you apply Module 4 in your capstone (**capstone brief**)

### [Capstone Project](/projects)

A multimodal agent using at least two GPU media models, shared publicly and running on Nebius Serverless AI. You write the proposal in Module 1 and submit the project by **Fri 15 Jan 2027**, then review your peers' projects.

<p align="center">
  <img src="images/capstone-requirements.png" alt="Capstone requirements: an agent with 2+ tools, an async GPU workload, assets in object storage, an eval suite, tracing, an SLO, a cost report, and proof that a killed run resumes without redoing finished work" width="100%">
</p>

Your project needs:

* An agent with 2+ tools
* An async GPU workload
* Assets in object storage
* An eval suite
* Tracing
* An SLO
* A cost report
* Proof that a killed run resumes without redoing finished work

Ideas: a marketing studio, podcast producer, video localization, storybook generator, product ad agent or paper-to-explainer video.

Share your progress as you go: see [Learning in public](learning-in-public.md).

## Who Should Join

Working builders running open or custom GPU media models:

* AI/ML engineers adding image, video or audio generation to real systems
* MLOps and ML platform engineers who own the batch pipeline, scheduling and cost line
* Technical founders and small teams paying for per-call media APIs who need to know when self-hosting wins

**Level:** intermediate to advanced.

## Prerequisites

* Python
* Docker basics
* HTTP APIs
* Command line

Prior LLM or MLOps course experience helps but isn't required.

**Not covered:** diffusion maths, training foundation models, deep fine-tuning, CUDA, Kubernetes, RAG, LLM fundamentals.

## How to Join

The course is free. Seats in the live cohort are limited, so it is application only.

1. [Apply here](https://courses.datatalks.club/register/production-ai-agents-generative-media/) and tell us what you are building.
2. We review every application, usually within 2 business days.
3. Approved applicants join the live cohort: shared deadlines, peer-reviewed homework and the capstone.
4. Get GPU credits through the [Nebius AI Builder Program](https://dev.nebius.com/builders).

"Live cohort" means shared deadlines and peer-reviewed homework, not mandatory live classes. Lectures are pre-recorded, and there is a weekly live office hour.

You can apply after the course starts on Mon 16 Nov: Module 1 runs two weeks, so you have time to catch up.

Not applying? All materials are in this repo and anyone can follow along. You'll need your own GPU access to run the code.

## Cost

The course itself is free. The workloads run on GPUs, which cost money:

* **GPU credits:** applicants get credits through the [Nebius AI Builder Program](https://dev.nebius.com/builders), which is free to join. Usage beyond the credits is billed to your account.
* **Following on your own:** you pay for your own GPU usage.

Each module reports cost per asset, so you'll know what a run costs before you scale it.

## What You Will Use

| Layer | Course default | Alternatives |
|---|---|---|
| Media runtimes | Hugging Face Diffusers, model-specific runtimes | ComfyUI for prototyping |
| Models | Image: Qwen-Image / FLUX-class, Sana · Video: Wan · Audio: Kokoro / Chatterbox TTS, Whisper | Any open-weights equivalent |
| LLM | Nebius Token Factory (OpenAI-compatible) | Self-hosted vLLM, any OpenAI-compatible API |
| Agents and tools | Raw tool calling → MCP | Pydantic AI, LangGraph |
| GPU compute | Nebius Serverless AI: Jobs, Endpoints | DevLabs with JupyterLab |
| Storage and API | S3-compatible Object Storage · FastAPI | Gradio, Streamlit |
| Tracking, tracing, evals | To be announced | Langfuse, OpenTelemetry |

## Open and Portable

* **License:** MIT
* **Portable code:** OpenAI- and S3-compatible, so you can run it elsewhere
* **Reference platform:** Nebius Serverless AI (Jobs, Endpoints) and Nebius Token Factory
* **Model licences:** open-weights models come with their own licences. Check before commercial use.

## Course Creators

<p align="center">
  <img src="images/partners-light.png" alt="Production AI Agents for Generative Media: build, scale and monitor agents that run image, video and audio models on serverless GPUs. Nebius × DataTalks.Club" width="100%">
</p>

* [DataTalks.Club](https://datatalks.club/): a community of data enthusiasts who learn and grow together. We share knowledge, help each other, and make data science more accessible.
* [Nebius](https://nebius.com/): the Nebius DevRel team creates and teaches the course.
* [Nebius Academy](https://nebius.com/certification): content and expert support.
* [Nebius Serverless AI](https://nebius.com/serverless): the reference platform for code-alongs and homework. Run containerized AI workloads on GPUs without managing infrastructure.
* [Nebius AI Builder Program](https://dev.nebius.com/builders): how applicants get GPU credits. Free to join: credits, runnable examples and cookbooks, technical office hours and a community of AI builders.

People behind this course:
* [Alexey Grigorev](https://linkedin.com/in/agrigorev): the founder of DataTalks.Club and the creator of its free courses.
* [Mikhail Rozhkov](https://www.linkedin.com/in/mnrozhkov/): a Senior Product Manager for Serverless AI at Nebius, where he took Serverless AI (GPU Jobs and Endpoints) from concept to public launch. He is an AI/ML engineer with a background in MLOps and robotics, and he co-leads the Nebius side of the course "Production AI Agents for Generative Media".
* [Marouane Khoukh](https://linkedin.com/in/marouanekhoukh/): AI Engineer, DevRel at Nebius.
* [Aleksandr Iziumov](https://linkedin.com/in/aiziumov/): Staff Customer Engineer, AI & Agentic at Nebius.

## FAQ

**Do I need my own GPU?**
No GPU or cluster to manage: the models run as Jobs and Endpoints on Nebius Serverless AI. Applicants get GPU credits through the [Nebius AI Builder Program](https://dev.nebius.com/builders); usage beyond the credits is billed to your account. If you follow along without applying, you need your own GPU access.

**Can I join after the course starts?**
Yes. Module 1 runs two weeks, so late joiners can catch up.

**Do I have to attend live sessions?**
No. Lectures are pre-recorded and live classes aren't mandatory. The pre-course Live Q&A, the course launch and the weekly office hour are live.

**What if I'm not approved?**
All materials are public (MIT), so you can follow along on your own, with your own GPU access.

**Am I locked into Nebius?**
No. The code is OpenAI- and S3-compatible, so you can run it elsewhere. Nebius Serverless AI and Nebius Token Factory are the reference platform for the code-alongs and homework.

**Can I use the generated media commercially?**
Open-weights models come with their own licences. Check each model's licence before commercial use.

## Getting Help

Ask questions in the [DataTalks.Club Slack](https://datatalks.club/slack.html) and bring them to the weekly live office hour.
