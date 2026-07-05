# Label-noise robustness experiments

This repository evaluates semantic-segmentation metrics under seven synthetic
label-noise mechanisms using both model predictions and a perfect predictor.

## Reproduce the experiment databases

1. Install the locked environment:

   ```bash
   uv sync
   ```

2. Place the OpenEarthMap-SAR test split below `data/oem_sar/test/` and ensure
   the 490 prediction masks are available in `data/oem_sar/results/`.

3. Review `config/experiments.toml`. Model inference is optional and disabled
   by default.

4. Open and run `src/run_experiments.ipynb` from top to bottom.

The notebook constructs exactly 350 canonical noise datasets and produces:

- `experiments/eval_results.db`
- `experiments/eval_results_perfect_pred.db`
- `experiments/experiment_manifest.json`

Boundary IoU is calculated per foreground class by summing boundary
intersections and unions across the 490 images before division. Boundaries are
still extracted within each image, so they never cross image edges.

Execution is deterministic and resumable. Existing databases are backed up
before canonical replacements are activated, and generated noise directories
are never deleted automatically.

Generation and evaluation are separate, resumable phases. The same API used by
the notebook is available directly:

```python
from experiment_pipeline import generate_experiments, evaluate_experiments

generation = generate_experiments("config/experiments.toml")
evaluation = evaluate_experiments("config/experiments.toml")
```

`generate_experiments()` creates or adopts and verifies all noisy masks. It
does not calculate metrics. `evaluate_experiments()` consumes those masks,
calculates both predictor variants, validates the paired databases, and then
activates them. `run_experiment()` remains available as a compatibility wrapper
that invokes both phases in sequence.

When running this snippet outside the notebook, add `src/` to `PYTHONPATH`.

## Tests

```bash
uv run pytest -q
```
