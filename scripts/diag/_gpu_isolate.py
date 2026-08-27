"""Run a GPU workload in a throwaway subprocess so its memory is fully reclaimed.

vLLM v1 runs its engine in a separate `EngineCore` process. `del llm` drops the
handle in the parent but does not reliably reap that child, so a second model loaded
afterwards in the same script finds the GPU still occupied:

    ValueError: Free memory on device (10.26/79.27 GiB) on startup is less than
    desired GPU memory utilization (0.85, 67.38 GiB)

Rather than chase vLLM's cleanup semantics across versions, run each model in its own
process and let the OS reclaim everything when that process exits. Costs one process
spawn and a model reload per call, which is negligible next to generation time.
"""

import multiprocessing as mp
import traceback


def _worker(fn, kwargs, q):
    try:
        q.put(("ok", fn(**kwargs)))
    except Exception:
        q.put(("err", traceback.format_exc()))


def run_isolated(fn, **kwargs):
    """Call ``fn(**kwargs)`` in a spawned subprocess and return its result.

    ``fn``, ``kwargs`` and the return value must all be picklable, so ``fn`` has to be
    a module-level function. Re-raises any exception from the child as a RuntimeError
    carrying its traceback.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(fn, kwargs, q))
    p.start()
    # Drain the queue BEFORE joining: a child that puts a large object and then blocks
    # on the pipe would deadlock against a parent already waiting in join().
    status, payload = q.get()
    p.join()
    if status == "err":
        raise RuntimeError(f"isolated worker failed:\n{payload}")
    return payload
