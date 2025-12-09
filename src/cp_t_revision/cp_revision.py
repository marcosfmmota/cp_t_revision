from collections import defaultdict

import torch
import torchcp
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from torchcp.classification.predictor.base import BasePredictor

from cp_t_revision.models import ModelModule


def get_initial_transition_matrix(
    model: ModelModule,
    calibration_set: DataLoader,
    alpha: float,
    k_points: int,
    cp_score: str,
    cp_predictor: str,
) -> Tensor:
    score = getattr(torchcp.classification.score, cp_score)()
    predictor: BasePredictor = getattr(torchcp.classification.predictor, cp_predictor)(
        score, model=model
    )
    predictor.calibrate(calibration_set, alpha)
    cal_labels_pred_sets = defaultdict(list)
    # The `*_` is used to unpack and ignore any additional elements in the batch tuple beyond features and targets.
    for x_batch, label_batch, *_ in calibration_set:
        # Ensure model and inputs are on the same device
        device = next(model.parameters()).device
        x_batch = x_batch.to(device)

        pred_sets = predictor.predict(x_batch)
        len_pred_sets = [sum(pred_set) for pred_set in pred_sets.tolist()]
        batch_logits = model(x_batch)
        for len_pred_set, logits, label in zip(
            len_pred_sets, batch_logits.tolist(), label_batch.tolist()
        ):
            cal_labels_pred_sets[label].append((len_pred_set, logits))

    # Sort prediction set length in ascending order
    sorted_labels_logits = {
        k: sorted(v, key=lambda tup: tup[0]) for k, v in cal_labels_pred_sets.items()
    }

    # Sort dictionary key to ensure label encoding ordering
    sorted_labels_logits = {
        k: [tup[1] for tup in v]
        for k, v in sorted(sorted_labels_logits.items(), key=lambda item: item[0])
    }

    # Compute the mean of softmaxed logits for the k anchor points
    transition_matrix = [
        torch.mean(
            torch.stack([softmax(torch.tensor(logit), dim=0) for logit in v[:k_points]]), dim=0
        )
        for _, v in sorted_labels_logits.items()
    ]

    return torch.stack(transition_matrix)
