---
video_url: ""
prev_url: 01-evaluating-generative-systems.md
next_url: 03-observability-tracing.md
---
# 4.2 Judge Calibration and Regression Gates

In this lesson we measure how far the 2.5 judge agrees with our own labels, fix the rubric where it disagrees, and block a bad change with a golden set and a regression gate. You're done when `calibration.json` shows agreement for rubric v1 and v2 on your 40 labelled assets, and `gate.json` shows the gate failing a worse candidate.

## Recap: rubric v1 and the verdicts in your run folder

The rubric and the judge come from [2.5](../02-multi-agent-media-workflows/05-judge-loop.md). The labels come from [4.1](01-evaluating-generative-systems.md).

## Calibration: your labels against the judge

## Disagreements: false accepts first

## Multimodal failure modes you found

## Rubric v2: fix, then re-judge only what changed

## Held-out check

With 40 labels and 10 held out, one disagreement moves agreement by 10 points. Treat it as a check that the fix generalises, not as a precise score.

## Golden set

## Regression gate

## Practice: gate a worse candidate
