# A Conformal Loss Correction Loss Correction Approach for Label-Noise Learning

Repository for the paper "A Conformal Loss Correction Loss Correction Approach for Label-Noise Learning".

This project implements CP T-Revision, a method that uses conformal prediction to select better anchor points for loss correction methods in label-noise learning. The CP T-Revision approach improves robustness to noisy labels by using conformal prediction to choose trustworthy anchors points for loss correction methods and improve over T-Revision method.

Key points
- **Proposed method:** CP T-Revision
- **Main implementation:** The core logic of choosing anchor points based on conformal prediction from the CP T-Revision method is implemented in `src/cp_t_revision/cp_revision.py`.
- **Configuration:** Experiments are configured with Hydra using the YAML files under `src/cp_t_revision/configs/experiment/`.
- **Project management:** This repository uses `pixi` for project management (task running / experiment orchestration) alongside Hydra for experiment configuration.

Quick start

1. Set up a Python environment using pixi.

2. Install required packages. The project uses PyTorch, Hydra, among other dependencies on `pyproject.toml`.

3. Run experiments

Use the centralized experiment runner and provide Hydra configuration from `src/cp_t_revision/configs/experiment`. For example:

```
python src/cp_t_revision/run_experiment.py experiment=cifar10_resnet_cprevision
```

Pass arguments according to the YAML files in `src/cp_t_revision/configs/experiment/` to select dataset, model, noise type, and other training options. For more documentation check [Hydra](https://hydra.cc/docs/intro/)

Repository layout (important files)
- **`src/cp_t_revision/cp_revision.py`**: Main CP T-Revision logic and loss-correction selection.
- **`src/cp_t_revision/run_experiment.py`**: Experiment entrypoint; uses Hydra-configured arguments.
- **`src/cp_t_revision/configs/`**: Configuration tree (datasets, models, noise transforms, and experiment presets).
- **`src/cp_t_revision/models.py`**, **`losses.py`**, **`data_modules.py`**: Model, loss, and data loading utilities.
