#!/usr/bin/env python3
import sys, csv, re, json
from pathlib import Path
from typing import Any, List, Dict, Optional

GRAY = "\033[90m"
RESET = "\033[0m"
W = 40

COST_FIELDS = {"Operator costs", "Transpose costs (shared)", "Transpose costs",
                "Layout cost (shared)", "Layout cost", "Layout D2D cost"}
ONE_DECIMAL_FIELDS = {"Avg. branching", "Estimated HBM usage"}

def row(k, v):
    ks = k.strip()
    if isinstance(v, (int, float)) and ks in COST_FIELDS:
        v = f"{v:,.0f}"
    elif isinstance(v, (int, float)) and ks in ONE_DECIMAL_FIELDS:
        v = f"{v:.1f}"
    elif isinstance(v, (int, float)) and ("instr." in ks or "cost" in ks.lower()):
        v = f"{v:,.0f}"
    print(f"{GRAY}{k.ljust(W)}{RESET}: {v}")

def extract_tensorizer_stats(p, stats):
    patterns = {
        (m, s): re.compile(rf"\[([a-zA-Z0-9]+)/Tensorizer/Statistics\]:\s*([\d.]+)\s*{re.escape(m)}\s*{re.escape(s)}")
        for m, s, _ in stats
    }
    by_sg = {}
    with p.open('r', errors='ignore') as f:
        for line in f:
            if 'Tensorizer/Statistics' not in line:
                continue
            for m, s, n in stats:
                match = patterns[(m, s)].search(line)
                if match:
                    sg, v = match.group(1), match.group(2)
                    by_sg.setdefault(sg, {})[n] = int(v) if v.isdigit() else float(v)
                    break
    return by_sg

def cycles_sum(p):
    idx = None
    total = 0
    found = False
    with p.open('r', errors='ignore') as f:
        for line in f:
            s = line.lstrip()
            if s.startswith("|") and "NODE ID" in s and "CYCLES" in s:
                hdr = [c.strip() for c in s.strip().strip("|").split("|")]
                idx = hdr.index("CYCLES") if "CYCLES" in hdr else None
                continue
            if idx is not None and s.startswith("|"):
                cols = [c.strip() for c in s.strip().strip("|").split("|")]
                try:
                    int(cols[0])
                    total += int(cols[idx])
                    found = True
                except:
                    pass
    return total if found else None

TENSORIZER_STATS = []
TENSORIZER_STATS_PER_SUBGRAPH = [
    ("ParAxesAnnotation", "Number of operators", "num_operators"),
    ("ParAxesAnnotation", "Number of edges", "num_edges"),
    ("ParAxesAnnotation", "Number of instructions after layout", "Layout instr."),
    ("TilingProfiler", "Real number of all insts after tiling", "Tiling instr."),
    ("ParAxesAnnotation", "Layout operator cost", "Layout operator cost"),
    ("ParAxesAnnotation", "Layout transpose cost", "Layout transpose cost"),
    ("ParAxesAnnotation", "Layout (operator+transpose+IO) cost", "Layout (operator+transpose+IO) cost"),
    ("ParAxesAnnotation", "Layout DMA cost (load+store within operators)", "Layout DMA cost"),
    ("ParAxesAnnotation", "Layout D2D transpose cost (offloaded transposes for non-local tensors)", "Layout D2D cost"),
    ("ParAxesAnnotation", "Estimated layout transposes (SBUF + IO, best-effort)", "num_layout_transposes"),
    ("ParAxesAnnotation", "Estimated tensors set non-local", "estimated_non_local_tensors"),
    ("InsertIOTransposes", "Number of inserted IO transposes", "num_io_transposes"),
    ("InsertLocalTransposes", "Number of PF transposes", "num_pf_transposes"),
    ("InsertLocalTransposes", "Number of tensors set non-local", "num_nonlocal_tensors"),
]

LAYOUT_RESULT_PATTERNS = [
    (re.compile(r"\[([a-zA-Z0-9]+)/Tensorizer/LayoutResult\]:\s*Inserted\s+(\d+)\s+IO transposes"), "num_io_transposes"),
    (re.compile(r"\[([a-zA-Z0-9]+)/Tensorizer/LayoutResult\]:\s*Inserted\s+(\d+)\s+PF transposes"), "num_pf_transposes"),
    (re.compile(r"\[([a-zA-Z0-9]+)/Tensorizer/LayoutResult\]:\s*Inserted\s+(\d+)\s+non-local tensors"), "num_nonlocal_tensors"),
]

FIELDS = [
    "model", "algorithm", "error_code", "TpbSgCyclesSum", "Tensorizer (sec)",
    "Layout (sec)", "BackendDriver (sec)", "production_total (sec)", "maxrss (gb)",
    "subgraphs", "HLO instr.", "Layout instr.", "Tiling instr.", "Backend instr.",
    "num_operators", "num_edges", "Layout operator cost", "Layout transpose cost",
    "Layout (operator+transpose+IO) cost", "Layout DMA cost", "Layout D2D cost", "num_layout_transposes", "estimated_non_local_tensors",
    "num_pf_transposes", "num_nonlocal_tensors", "num_io_transposes",
]

def process_dir(base):  # type: (Path) -> Optional[List[Dict[str, Any]]]
    metrics = base / "all_metrics.csv"
    log = base / "log-neuron-cc.txt"
    infer = base / "infer_errors.txt"
    compile_log = base / "log-compile.txt"
    if not metrics.exists():
        return None
    print(f"\n{metrics.resolve()}")
    name = base.name[4:] if base.name.startswith("out-") else base.name
    parts = name.rsplit("_", 1)
    data = {"model": parts[0], "algorithm": parts[1] if len(parts) > 1 else ""}
    want = {
        "Tensorizer": "Tensorizer (sec)",
        "BackendDriver": "BackendDriver (sec)",
        "production_total": "production_total (sec)",
    }
    metric_json = base / "tensorizer_metric_store.json"
    paa_sec = None
    if metric_json.exists():
        try:
            paa_sec = round(json.loads(metric_json.read_text())["Sum"]["compiletime"]["ParAxesAnnotation"])
        except (KeyError, json.JSONDecodeError):
            pass
    cycles = None
    if infer.exists():
        cycles = cycles_sum(infer)
        row("TpbSgCyclesSum", f"{cycles:,}" if cycles is not None else "not found")
        data["TpbSgCyclesSum"] = cycles
    else:
        row("TpbSgCyclesSum", "not found")
        data["TpbSgCyclesSum"] = None
    seen = set()
    with metrics.open(newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("unit") or "").strip() != "Seconds":
                continue
            scope = (r.get("scope") or "").strip()
            sub_scope = (r.get("sub_scope") or "").strip()
            if sub_scope == "BackendDriver" and scope != "all":
                continue
            key = scope if scope in want else sub_scope if sub_scope in want else None
            if key and key not in seen:
                seen.add(key)
                v = round(float(r["value"]))
                row(want[key], v)
                data[want[key]] = v
                if key == "Tensorizer":
                    row("Layout (sec)", paa_sec if paa_sec is not None else "not found")
                    data["Layout (sec)"] = paa_sec
    maxrss = None
    hlo_insts = None
    total_insts = None
    errors = []
    error_code = 0
    if log.exists():
        with log.open('r', errors='ignore') as f:
            for line in f:
                m = re.search(r"ru_maxrss:\s*(\d+)mb", line)
                if m:
                    maxrss = round(int(m.group(1)) / 1024, 1)
                m = re.search(r"Post-Partition Histogram after graph level optimizations - Total HLO instructions:\s*(\d+)", line)
                if m:
                    hlo_insts = int(m.group(1))
                m = re.search(r"Total number of instructions:\s*(\d+)", line)
                if m:
                    total_insts = int(m.group(1))
                if "[ERROR]" in line or "[INTERNAL_ERROR]" in line:
                    errors.append(line.strip())
                    code_match = re.search(r'\[(ERROR|INTERNAL_ERROR)\]\s*\[([A-Z_0-9]+)\]', line)
                    if code_match and error_code == 0:
                        error_code = code_match.group(2)
        all_stats = TENSORIZER_STATS + TENSORIZER_STATS_PER_SUBGRAPH
        all_results = extract_tensorizer_stats(log, all_stats)
        sg_stats_per_subgraph = {}
        for sg, stats_dict in all_results.items():
            sg_stats_per_subgraph[sg] = {k: v for k, v in stats_dict.items() if any(k == n for _, _, n in TENSORIZER_STATS_PER_SUBGRAPH)}
        with log.open('r', errors='ignore') as f:
            for line in f:
                if 'Tensorizer/LayoutResult' not in line:
                    continue
                for pat, name in LAYOUT_RESULT_PATTERNS:
                    match = pat.search(line)
                    if match:
                        sg, v = match.group(1), int(match.group(2))
                        sg_stats_per_subgraph.setdefault(sg, {})[name] = v
                        break
    row("maxrss (gb)", maxrss if maxrss is not None else "not found")
    data["maxrss (gb)"] = maxrss
    row("HLO instr.", hlo_insts if hlo_insts is not None else "not found")
    data["HLO instr."] = hlo_insts
    for _, _, stat_name in [("ParAxesAnnotation", "Number of instructions after layout", "Layout instr."),
                             ("TilingProfiler", "Real number of all insts after tiling", "Tiling instr.")]:
        stat_sum = sum(sg.get(stat_name, 0) for sg in sg_stats_per_subgraph.values()) if sg_stats_per_subgraph else None
        row(stat_name, stat_sum if stat_sum is not None else "not found")
        data[stat_name] = stat_sum
    row("Backend instr.", total_insts if total_insts is not None else "not found")
    data["Backend instr."] = total_insts
    for _, _, stat_name in [("ParAxesAnnotation", "Number of operators", "num_operators"),
                             ("ParAxesAnnotation", "Number of edges", "num_edges")]:
        stat_sum = sum(sg.get(stat_name, 0) for sg in sg_stats_per_subgraph.values()) if sg_stats_per_subgraph else None
        row(stat_name, stat_sum if stat_sum is not None else "not found")
        data[stat_name] = stat_sum
    for _, _, stat_name in TENSORIZER_STATS_PER_SUBGRAPH:
        if stat_name in ["Layout instr.", "Tiling instr.", "num_operators", "num_edges"]:
            continue
        stat_sum = sum(sg.get(stat_name, 0) for sg in sg_stats_per_subgraph.values()) if sg_stats_per_subgraph else None
        row(stat_name, stat_sum if stat_sum is not None else "not found")
        data[stat_name] = stat_sum
    n = sum(1 for p in base.iterdir() if p.is_file() and re.match(r"^greedy\.py\.\d{6}$", p.name))
    if n == 0 and (base / "greedy.py").is_file():
        n = 1
    row("subgraphs", n if n else "not found")
    data["subgraphs"] = n if n else None
    if cycles is None and compile_log.exists():
        with compile_log.open('r', errors='ignore') as f:
            for line in f:
                if "inference failed" in line.lower() or "infer failed" in line.lower():
                    if error_code == 0:
                        error_code = "infer_failed"
                    errors.append("Inference failed (from log-compile.txt)")
                    break
    data["error_code"] = error_code
    if errors:
        print(f"\n{GRAY}Errors found:{RESET}")
        for err in errors:
            print(f"  {err}")
    return [data]

def main():
    build_dir = Path(".")  # Use current directory
    if len(sys.argv) < 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <regex>")
    pat = re.compile(sys.argv[1])
    dirs = sorted(d for d in build_dir.glob("out-*") if pat.search(d.name[4:] if d.name.startswith("out-") else d.name))
    results = []
    for d in dirs:
        if d.is_dir():
            rows = process_dir(d)
            if rows is None:
                continue
            results.extend(rows)
    if not results:
        raise SystemExit("no out-* directories found")
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)
    def model_sort_key(model_name):
        rows = by_model[model_name]
        all_success = all(r.get("error_code", 0) == 0 for r in rows)
        cycles_sum = sum(r.get("TpbSgCyclesSum") or 0 for r in rows)
        return (not all_success, cycles_sum)
    sorted_models = sorted(by_model.keys(), key=model_sort_key)
    sorted_results = []
    for model in sorted_models:
        model_rows = sorted(by_model[model], key=lambda x: x.get("algorithm", "").lower())
        sorted_results.extend(model_rows)
    out = Path("e2e_metrics.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(sorted_results)
    print(f"\nWrote {len(sorted_results)} rows to {out}")

if __name__ == "__main__":
    main()
