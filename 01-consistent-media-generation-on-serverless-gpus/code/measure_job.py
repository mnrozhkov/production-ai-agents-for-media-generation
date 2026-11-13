"""Attach observed lifecycle intervals (created, running, finished) to a metrics.json file."""
import argparse
import json
from pathlib import Path
from common import emit, interval, write_json


def enrich(m, created, running, finished):
    result = dict(m)
    result["startup_s"] = interval(created, running)
    interval(running, finished)
    result["billable_s"] = interval(created, finished)
    result["billing_basis"] = "create_to_terminal_wall_clock_proxy_not_invoice"
    result["lifecycle"] = {
        "created_at": created,
        "running_at": running,
        "finished_at": finished,
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metrics", type=Path)
    p.add_argument("--created-at", required=True)
    p.add_argument("--running-at", required=True)
    p.add_argument("--finished-at", required=True)
    a = p.parse_args()
    result = enrich(
        json.loads(a.metrics.read_text()),
        a.created_at,
        a.running_at,
        a.finished_at,
    )
    emit(a.metrics.parent, "lifecycle.recorded", billing_basis=result["billing_basis"])
    write_json(a.metrics, result)


if __name__ == "__main__":
    main()
