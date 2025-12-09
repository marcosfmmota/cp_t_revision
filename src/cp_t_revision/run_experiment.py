import hydra
import mlflow
import numpy as np
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import DictConfig

from cp_t_revision.cp_revision import get_initial_transition_matrix
from cp_t_revision.models import ModelModule, RevisionModelModule


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_experiment(cfg: DictConfig):
    exp_config = cfg["experiment"]
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
            experiment_tags |= {"dataset": type(datamodule).__name__}

        experiment_name = f"{experiment_tags['dataset']}_{experiment_tags['model']}/"
        if exp_config["cp_revision"]:
            experiment_name = (
                f"{experiment_tags['dataset']}_{experiment_tags['model']}_cp_revision/"
            )
            experiment_tags |= {
                "cp_score": exp_config["cp_score"],
                "cp_predictor": exp_config["cp_predictor"],
                "alpha": exp_config["alpha"],
                "k_points": exp_config["k_points"],
            }

        # Ugly but necessary to maintain compatibility with previous directory structure of experiments
        if exp_config["synthetic_noise"]:
            experiment_name += f"{experiment_tags['noise_type']}/{experiment_tags['noise_level']}/k_points_{exp_config['k_points']}"

        mlflow.pytorch.autolog(extra_tags=experiment_tags)  # type: ignore
        logger = instantiate(exp_config["logger"], name=experiment_name)
        trainer = instantiate(exp_config["trainer"], logger=logger, _convert_="all")

        if exp_config["checkpoint_path"] is None:
            model_module = ModelModule(model, loss, metrics, optimizer_cfg)
            trainer.fit(model_module, datamodule)
            trainer.test(model_module, datamodule, ckpt_path="best")
        elif exp_config["checkpoint_path"] is not None and not exp_config["cp_revision"]:
            model_module = ModelModule.load_from_checkpoint(
                exp_config["checkpoint_path"],
                model=model,
                loss=loss,
                metric_collection=metrics,
                optimizer_cfg=optimizer_cfg,
            )
            trainer.fit(model_module, datamodule)
            trainer.test(model_module, datamodule, ckpt_path="best")
        else:
            model_module = ModelModule.load_from_checkpoint(
                exp_config["checkpoint_path"],
                model=model,
                loss=loss,
                metric_collection=metrics,
                optimizer_cfg=optimizer_cfg,
            )

        if exp_config["cp_revision"]:
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
            transition_matrix.to("cuda")
            revision_loss = instantiate(exp_config["revision_loss"], T=transition_matrix)
            second_step_optimizer_config = exp_config.get("second_step_optimizer_config")
            revision_model = RevisionModelModule(
                model_module.model,
                revision_loss,
                metrics,
                second_step_optimizer_config,
            )
            trainer.fit(revision_model, datamodule)
            trainer.test(revision_model, datamodule, ckpt_path="best")
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
