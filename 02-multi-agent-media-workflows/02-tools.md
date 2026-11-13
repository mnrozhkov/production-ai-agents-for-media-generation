---
video_url: ""
prev_url: 01-agent-loop.md
next_url: 03-gpu-tools.md
---
# 2.2 Tools and Structured Outputs

In this lesson we design typed tools: the signature becomes the schema, retries are split into three kinds, and an idempotency key keeps us from paying for the same GPU work twice. You're done when the model fixes an invalid call after one validation retry, and a second identical `generate_image` call is a cache hit with no GPU time and no cost.

## The tool signature is the schema

## Three kinds of retry

## Tool timeouts

## Idempotency keys

## Return references, not bytes

## Structured outputs for LLM steps
