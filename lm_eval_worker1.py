"""
lm_eval_worker.py  (patched)

What changed vs your original file, and why:
  - A failing task no longer silently disappears. Any exception from
    evaluator.simple_evaluate() is now caught, logged with a full traceback,
    and recorded in the output JSON under "errors" instead of crashing the
    whole worker process. Real crashes get swallowed by the caller-side bug
    described in run_lm_eval_v2.py, so this is the other half of that fix.
  - GPU memory is printed before/after each task, so a silent OOM /
    CPU-fallback (which would explain DoRA's much longer runtimes) is
    visible in the logs instead of invisible.
  - Exit code 2 = "some tasks worked, some didn't" (partial success, still
    useful) vs exit code 1 = total crash before anything ran.
"""
import sys, json, os, traceback, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

from lm_eval import evaluator
from collections import defaultdict


def gpu_mem_snapshot():
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB"
    except Exception:
        pass
    return "n/a"


def main():
    cfg_path    = sys.argv[1]
    result_path = sys.argv[2]

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    base_model_id = cfg["base_model_id"]
    adapter_dir   = cfg["adapter_dir"]
    method_id     = cfg["method_id"]
    tasks         = cfg["tasks"]
    device        = cfg["device"]
    limit         = cfg.get("limit")

    fewshot_groups = defaultdict(list)
    for t in tasks:
        fewshot_groups[t["fewshot"]].append(t["name"])

    if method_id in {"SFT", "QAT_LORA"}:
        model_args_dict = {
            "pretrained": adapter_dir,
            "trust_remote_code": True,
            "dtype": "float16",
            "max_length": 1024,
        }
    else:
        model_args_dict = {
            "pretrained": base_model_id,
            "peft": adapter_dir,
            "trust_remote_code": True,
            "dtype": "float16",
            "max_length": 1024,
        }

    model_args = ",".join(f"{k}={v}" for k, v in model_args_dict.items())
    all_results = {}
    errors = {}

    print(f"  [worker] method={method_id} adapter={adapter_dir}", flush=True)
    print(f"  [worker] GPU before load: {gpu_mem_snapshot()}", flush=True)

    for fewshot, task_group in fewshot_groups.items():
        print(f"  [worker] Evaluating tasks={task_group} fewshot={fewshot}", flush=True)

        eval_kwargs = {
            "model": "hf",
            "model_args": model_args,
            "tasks": task_group,
            "device": device,
            "batch_size": "auto:4",
        }
        if fewshot is not None:
            eval_kwargs["num_fewshot"] = fewshot
        if limit is not None:
            eval_kwargs["limit"] = limit

        t0 = time.time()
        try:
            results = evaluator.simple_evaluate(**eval_kwargs)
            for task_name, task_result in results.get("results", {}).items():
                all_results[task_name] = {"results": {task_name: task_result}}
            print(f"  [worker] OK tasks={task_group} in {time.time()-t0:.1f}s "
                  f"GPU after: {gpu_mem_snapshot()}", flush=True)
        except Exception as e:
            # This is the important bit: don't let one bad task kill the
            # whole worker silently. Record what actually happened.
            tb = traceback.format_exc()
            print(f"  [worker] FAILED tasks={task_group}: {e}", flush=True)
            print(tb, flush=True)
            for task_name in task_group:
                errors[task_name] = {"error": str(e), "traceback": tb}

    output = {"results": all_results, "errors": errors}
    with open(result_path, "w") as f:
        json.dump(output, f)

    print(f"  [worker] Done. {len(all_results)} ok, {len(errors)} failed. "
          f"Results written to {result_path}", flush=True)

    if not all_results and errors:
        sys.exit(2)  # total failure, but errors are still on disk for inspection


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
