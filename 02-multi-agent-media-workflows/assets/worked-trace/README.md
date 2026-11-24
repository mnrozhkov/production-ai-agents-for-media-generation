# Worked trace

`events.jsonl` is a hand-written trace of one full run: plan, a keyframe that gets `RETRY`,
a second keyframe that gets `PASS`, one video, checks. `expected.json` holds the totals the
aggregation rules give for it. `check()` in `code/tracing.py` must pass on it:

```python
from tracing import check, read_events, summarize

events = read_events("../assets/worked-trace/events.jsonl")
assert check(events) == []
print(summarize(events))
```

`code/nb-agent-loop.ipynb` loads it in its first section.

All numbers are [illustrative]: invented to show how the totals add up, not measured.
