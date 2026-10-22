"""Compare validation/output on identical proposals from a real planner search.

Run with PYTHONPATH=. and the same thread settings as benchmark_coverage_planning.py.
The baseline modules are read from a local Git revision; no checkout is modified.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from benchmark_coverage_planning import Case, dataset, environment, measure

from deepcave.evaluators import coverage_planning as current


def baseline(ref):
    """Load the two pre-optimization modules without changing installed imports."""
    modules, hashes = {}, {}
    for name in ("combinatorial", "coverage_planning"):
        source = subprocess.check_output(["git", "show", f"{ref}:deepcave/evaluators/{name}.py"])
        module = ModuleType(f"_validation_baseline_{name}")
        exec(compile(source, f"{ref}/{name}.py", "exec"), module.__dict__)
        modules[name] = module
        hashes[name] = hashlib.sha256(source).hexdigest()
    engine = modules["coverage_planning"]
    engine.CoverageSpec = modules["combinatorial"].CoverageSpec
    return engine, hashes


def main():
    """Capture one real search, then compare postprocessing on fixed proposal prefixes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=8303)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    old, hashes = baseline(args.baseline_ref)
    case = Case(12, 8, 4)
    data = dataset(case)
    points = []
    state_type = old.CoverageState
    calls = 0

    def capture(*positional, **keywords):
        nonlocal calls
        calls += 1
        state = state_type(*positional, **keywords)
        if calls == 1:
            add = state.add

            def record(point):
                points.append(point.copy())
                return add(point)

            state.add = record
        return state

    # The initial real search is excluded from the paired postprocessing measurements.
    import benchmark_coverage_planning as benchmark

    with (
        patch.object(benchmark, "planner", old),
        patch.object(old, "MAX_ADDITIONAL_TRIALS", args.trials),
        patch.object(old, "CoverageState", capture),
    ):
        search = measure(data, case, 0, True, 120)
    assert len(points) == args.trials and not search["complete"]
    print(f"Captured {len(points)} proposals from a real search", flush=True)
    result = {
        "environment": environment(),
        "baseline_ref": args.baseline_ref,
        "baseline_source_sha256": hashes,
        "method": "Fixed real search proposals; compare validation_and_output_seconds only",
        "search": search,
        "proposal_sha256": hashlib.sha256(b"".join(p.tobytes() for p in points)).hexdigest(),
        "runs": [],
    }
    for count in dict.fromkeys((1000, args.trials)):
        reference_id = None
        for repeat in range(args.repeats):
            order = (("before", old), ("after", current))
            for label, engine in (order if repeat % 2 == 0 else reversed(order)):
                proposals = iter(points[:count])

                def candidates(spec, state, rng):
                    return next(proposals)[None, :]

                with (
                    patch.object(benchmark, "planner", engine),
                    patch.object(engine, "MAX_ADDITIONAL_TRIALS", count),
                    patch.object(engine, "_candidates", candidates),
                    patch.object(engine, "_improve", lambda point, state, rng, deadline: point),
                ):
                    record = measure(data, case, 0, True, 120)
                assert record["proposed_trials"] == count
                if reference_id is None:
                    reference_id = record["plan_id"]
                assert record["plan_id"] == reference_id, "Changed output for identical proposals"
                if count == args.trials:
                    assert record["plan_id"] == search["plan_id"]
                record.update(implementation=label, repetition=repeat)
                result["runs"].append(record)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(
                    f"{count} trials {label} repeat={repeat}: "
                    f"validation/output={record['validation_and_output_seconds']:.3f}s",
                    flush=True,
                )
    result["summary"] = [
        {
            "trials": count,
            "implementation": label,
            "median_validation_and_output_seconds": statistics.median(
                r["validation_and_output_seconds"]
                for r in result["runs"]
                if r["proposed_trials"] == count and r["implementation"] == label
            ),
        }
        for count in dict.fromkeys((1000, args.trials))
        for label in ("before", "after")
    ]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
