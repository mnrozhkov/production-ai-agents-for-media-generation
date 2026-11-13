"""Record Endpoint startup, readiness wait, and per-request latency into metrics.json."""
import argparse
from call_endpoint import run
from common import DEFAULT_CONFIG, positive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--n", type=positive, default=1)
    p.add_argument("--out")
    p.add_argument("--created-at", required=True, help="ISO timestamp with timezone, before Create")
    p.add_argument("--running-at", required=True, help="ISO timestamp with timezone, first RUNNING")
    a = p.parse_args()
    run(a.config, a.n, a.out, a.created_at, a.running_at)


if __name__ == "__main__":
    main()
