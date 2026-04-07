# T7 — Impact Persistence (Decay)

**T7 in the codebase corresponds to T5 in the paper.**

The original data construction pipeline uses `task5+7/` directories and `t7_` prefixes throughout. In the paper (ACM MM '26) and the public benchmark, this task is numbered **T5**.

All baselines and evaluation scripts for this task live in [`baselines/t5/`](../t5/) and are accessed via:

```python
from eventxbench import load_task

train, test = load_task("t5")
```

See [`baselines/t5/`](../t5/) for the full set of baselines.
