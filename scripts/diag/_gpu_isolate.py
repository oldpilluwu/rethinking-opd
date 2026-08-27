"""Run a GPU workload in a throwaway subprocess so its memory is fully reclaimed.

vLLM v1 runs its engine in a separate `EngineCore` process. `del llm` drops the
handle in the parent but does not reliably reap that child, so a second model loaded
afterwards in the same script finds the GPU still occupied:

    ValueError: Free memory on device (10.26/79.27 GiB) on startup is less than
    desired GPU memory utilization (0.85, 67.38 GiB)

Rather than chase vLLM's cleanup semantics across versions, run each model in its own
process and let the OS reclaim everything when that process exits.

The child can die WITHOUT putting anything on the queue -- most memorably via a stray
SIGALRM from the math grader's nested timeout machinery, whose default disposition
terminates the process silently. A bare `q.get()` then blocks the parent forever. So
this polls the queue and the child's liveness together, and turns a dead child into an
exception the caller can handle instead of a hang.
"""

import multiprocessing as mp
import queue as _queue
import traceback

_POLL_SECONDS = 5.0
# Grace period after the child exits, so a result already in flight through the pipe
# is not mistaken for a silent death.
_DRAIN_SECONDS = 10.0


def _worker(fn, kwargs, q):
    try:
        q.put(("ok", fn(**kwargs)))
    except BaseException:  # BaseException: SystemExit/KeyboardInterrupt should report too
        q.put(("err", traceback.format_exc()))


def run_isolated(fn, **kwargs):
    """Call ``fn(**kwargs)`` in a spawned subprocess and return its result.

    ``fn``, ``kwargs`` and the return value must all be picklable, so ``fn`` has to be
    a module-level function. Raises RuntimeError if the child fails or dies without
    producing a result -- never blocks indefinitely.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(fn, kwargs, q))
    p.start()

    result = None
    drained = 0.0
    while True:
        try:
            result = q.get(timeout=_POLL_SECONDS)
            break
        except _queue.Empty:
            if p.is_alive():
                continue
            # Child is gone. Give the pipe a moment in case a result is still landing.
            drained += _POLL_SECONDS
            if drained >= _DRAIN_SECONDS:
                break

    p.join(timeout=30)
    if p.is_alive():
        p.terminate()
        p.join(timeout=10)

    if result is None:
        raise RuntimeError(
            f"isolated worker died without returning a result (exit code {p.exitcode}). "
            f"A negative code is a signal -- e.g. -14 is SIGALRM, which the math grader's "
            f"nested timeouts can raise with no handler installed."
        )
    status, payload = result
    if status == "err":
        raise RuntimeError(f"isolated worker failed:\n{payload}")
    return payload
