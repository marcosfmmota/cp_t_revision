import hydra
import mlflow
import numpy as np
import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

from cp_t_revision.cp_revision import get_initial_transition_matrix
from cp_t_revision.models import ModelModule, RevisionModelModule


def _build_trainer(trainer_cfg, logger, monitor: str):
    """Instantiate a Trainer with a fresh ModelCheckpoint so ``ckpt_path="best"`` resolves
    to the monitored validation metric and callback state does not leak between stages.

    A ModelCheckpoint is injected only when the config does not already declare callbacks,
    so user-supplied callbacks take precedence and there is no ambiguous duplicate "best".
    Passing ``callbacks`` as a kwarg would override any callbacks in the config, so when the
    config already declares callbacks we let Hydra instantiate them as-is.
    """
    if trainer_cfg.get("callbacks"):
        return instantiate(trainer_cfg, logger=logger, _convert_="all")
    return instantiate(
        trainer_cfg,
        logger=logger,
        callbacks=[ModelCheckpoint(monitor=monitor, mode="max", save_top_k=1)],
        _convert_="all",
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_experiment(cfg: DictConfig):
    exp_config = cfg["experiment"]

    # Configure MLFlow once: point the client at a tracking server when configured, and enable
    # autologging. Per-run names/experiments are set inside the loop below. When tracking_uri is
    # null the MLFLOW_TRACKING_URI env var (if set) is honored, otherwise a local ./mlruns store is used.
    mlflow_cfg = cfg.get("mlflow") or {}
    tracking_uri = mlflow_cfg.get("tracking_uri")
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.pytorch.autolog(log_models=mlflow_cfg.get("log_models", False))

    for run in range(exp_config["n_runs"]):
        # Change seed based on run to have variability between runs not between experiments
        seed_everything(exp_config["seed"] + run)
        model = instantiate(exp_config["model"]["model"], _convert_="all")
        loss = instantiate(exp_config["model"]["loss"], _convert_="all")
        metrics = instantiate(exp_config["model"]["metric_collection"], _convert_="all")
        optimizer_cfg = exp_config["model"]["optimizer_cfg"]
        experiment_tags = {"model": type(model).__name__}
        if exp_config["synthetic_noise"]:
            noise = instantiate(exp_config["noise_transform"], n_classes=exp_config["n_classes"])
            datamodule = instantiate(exp_config["dataset"], noise_transform=noise)
            experiment_tags |= {
                "dataset": type(datamodule).__name__,
                "noise_type": type(datamodule.noise_transform).__name__,
                "noise_level": datamodule.noise_transform.noise_level,
            }
        else:
            datamodule = instantiate(exp_config["dataset"])
            experiment_tags |= {"dataset": type(datamodule).__name__, "noise_type": exp_config["dataset"].get("noise_type", "none")} # only works with CIFAR-N dataset, for other datasets it will be "none"

        experiment_name = f"{experiment_tags['dataset']}_{experiment_tags['model']}/"
        if exp_config["cp_revision"]:
            experiment_name = (
                f"{experiment_tags['dataset']}_{experiment_tags['model']}_cp_revision/{exp_config['cp_score']}/k_points_{exp_config['k_points']}/alpha_{exp_config['alpha']}/"
            )
            experiment_tags |= {
                "cp_score": exp_config["cp_score"],
                "cp_predictor": exp_config["cp_predictor"],
                "alpha": exp_config["alpha"],
                "k_points": exp_config["k_points"],
            }
            if not exp_config["synthetic_noise"]:
                experiment_name += f"{experiment_tags['noise_type']}/"

        # Ugly but necessary to maintain compatibility with previous directory structure of experiments
        if exp_config["synthetic_noise"]:
            experiment_name += f"{experiment_tags['noise_type']}/{experiment_tags['noise_level']}"

        # MLFlow names cannot contain slashes (the CSVLogger path above relies on them), so derive a
        # slash-free experiment name and a unique run name per loop iteration.
        mlflow_experiment = experiment_name.strip("/").replace("/", "_")
        run_name = f"{mlflow_experiment}_run{run}_seed{exp_config['seed'] + run}"
        mlflow.set_experiment(mlflow_experiment)

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(experiment_tags)
            mlflow.set_tag("run_index", str(run))
            mlflow.set_tag("seed", str(exp_config["seed"] + run))

            monitor = exp_config.get("checkpoint_monitor", "val_MulticlassAccuracy")

            # Base stage: train from scratch (no checkpoint) or fine-tune from a checkpoint. It is
            # skipped when a checkpoint is loaded purely to seed CP revision (no base trainer built).
            checkpoint_path = exp_config["checkpoint_path"]
            run_base = checkpoint_path is None or not exp_config["cp_revision"]

            base_trainer = None
            if run_base:
                # The base logger is constructed and its fit run before the revision logger below,
                # so the revision CSVLogger picks an incremented version rather than colliding.
                base_logger = instantiate(exp_config["logger"], name=experiment_name)
                base_trainer = _build_trainer(exp_config["trainer"], base_logger, monitor)
                if checkpoint_path is None:
                    model_module = ModelModule(model, loss, metrics, optimizer_cfg)
                else:
                    model_module = ModelModule.load_from_checkpoint(
                        checkpoint_path,
                        model=model,
                        loss=loss,
                        metric_collection=metrics,
                        optimizer_cfg=optimizer_cfg,
                    )
                base_trainer.fit(model_module, datamodule)
                base_trainer.test(model_module, datamodule, ckpt_path="best")
            else:
                # Checkpoint loaded purely for CP revision: no base training.
                model_module = ModelModule.load_from_checkpoint(
                    checkpoint_path,
                    model=model,
                    loss=loss,
                    metric_collection=metrics,
                    optimizer_cfg=optimizer_cfg,
                )

            if exp_config["cp_revision"]:
                # Seed revision from the best base checkpoint (when base training ran) so the
                # transition matrix and revision init use the best-validation weights rather than
                # the final-epoch weights left in memory after fit.
                if base_trainer is not None and exp_config.get("revision_from_best_ckpt", True):
                    best_path = getattr(base_trainer.checkpoint_callback, "best_model_path", "")
                    if best_path:
                        # weights_only=False because Lightning checkpoints also store OmegaConf
                        # hyper_parameters; the file is trusted (written by our own trainer above).
                        state = torch.load(best_path, map_location="cpu", weights_only=False)
                        model_module.load_state_dict(state["state_dict"])

                # Fresh logger + trainer for revision: independent max_epochs and clean callback
                # state, so ckpt_path="best" resolves to this stage's own ModelCheckpoint.
                revision_logger = instantiate(exp_config["logger"], name=experiment_name)
                revision_trainer = _build_trainer(
                    exp_config.get("revision_trainer", exp_config["trainer"]),
                    revision_logger,
                    monitor,
                )

                datamodule.setup(stage="fit")
                calibration_set = datamodule.val_dataloader()
                transition_matrix = get_initial_transition_matrix(
                    model_module,
                    calibration_set,
                    exp_config["alpha"],
                    exp_config["k_points"],
                    exp_config["cp_score"],
                    exp_config["cp_predictor"],
                )
                np.save("initial_transition_matrix.npy", transition_matrix.cpu().numpy())
                mlflow.log_artifact("initial_transition_matrix.npy", artifact_path="transition_matrix")
                device = next(model_module.parameters()).device
                transition_matrix = transition_matrix.to(device)
                revision_loss = instantiate(exp_config["revision_loss"], T=transition_matrix)
                second_step_optimizer_config = exp_config.get("second_step_optimizer_config")
                revision_model = RevisionModelModule(
                    model_module.model,
                    revision_loss,
                    metrics,
                    second_step_optimizer_config,
                )
                revision_trainer.fit(revision_model, datamodule)
                revision_trainer.test(revision_model, datamodule, ckpt_path=exp_config["ckpt_strategy"])
                np.save(
                    "final_transition_matrix.npy",
                    revision_model.T_matrix.cpu().numpy(),  # type: ignore
                )
                mlflow.log_artifact("final_transition_matrix.npy", artifact_path="transition_matrix")
                np.save(
                    "final_correction.npy",
                    revision_model.noise_adaptation.weight.detach().cpu().numpy(),  # type: ignore
                )
                mlflow.log_artifact("final_correction.npy", artifact_path="transition_matrix")


if __name__ == "__main__":
    run_experiment()
