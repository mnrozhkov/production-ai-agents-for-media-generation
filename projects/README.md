# Capstone Project: Production Multimodal Agent

Build a production multimodal agent and share it publicly. The capstone starts in Module 1 and has its first submission deadline on **15 Jan 2027**.

## Requirements

* A production multimodal agent that uses **at least 2 GPU media models** (any combination of image, video, audio)
* An **async workload**: the agent orchestrates long-running GPU jobs (job IDs, state persistence, resume — not blocking calls)
* **Batch run**: 100 records for image and audio assets, 10 records for video assets
* **Evaluation**: a golden set plus a regression gate, at system, agent and media layers
* **Observability**: a trace per request that answers "why this cost / why this long", plus metrics, logs and a dashboard
* Run it on **Nebius Serverless AI** (Jobs, Endpoints) as the reference execution layer
* Code is open-source (MIT), OpenAI- and S3-compatible

## Records

Bring your own records: a CSV or JSONL file with an id column. Records go in, generated media comes out.

## Review

* Capstones are **peer-reviewed** against an 8-area rubric (covering the system, agent and media layers plus cost, reliability, observability, reproducibility and presentation)
* To pass, submit your project, review peers, and share your work publicly

## Timeline

* Start: Module 1 (two weeks, Mon 16 – Sun 29 Nov 2026) — pick your records, get the first GPU generation working, and fill in the [Capstone Proposal Template](proposal-template.md)
* Modules 2–4 (weekly): Mon 30 Nov – Sun 20 Dec 2026
* Teaching ends: Sun 20 Dec 2026
* Attempt 1: submit by 15 Jan 2027; complete peer reviews by 22 Jan 2027
* Attempt 2: submit by 29 Jan 2027; complete final peer reviews by 5 Feb 2027
* Submission and peer-review deadlines are 23:59:59 UTC on the dates above.

(More details coming soon)
