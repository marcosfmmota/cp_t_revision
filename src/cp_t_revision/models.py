import torch
import torch.nn.functional as F
import torchvision
from lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MetricCollection
from transformers import XLMRobertaForSequenceClassification

from cp_t_revision.losses import ReweightCorrectionLoss, Reweighting_Revision_Loss


class LeNet5(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=400, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=self.n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.pool1(self.relu(self.conv1(x)))
        out = self.pool2(self.relu(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        resnet_type: str = "resnet18",
        weights=None,
    ) -> None:
        super().__init__()
        self.model = getattr(torchvision.models, resnet_type)(weights=weights)
        # freeze weights when doing fine-tuning
        if weights:
            for param in self.model.parameters():
                param.requires_grad = False
        # Adapting the first conv layer to the case of smaller image sizes (like CIFAR 10 and 100)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.maxpool = nn.Identity()
        num_filters = self.model.fc.in_features
        self.n_classes = n_classes
        self.model.fc = nn.Linear(num_filters, self.n_classes)

    def forward(self, x: Tensor):
        if x.shape[1] != 3:  # In the case datasets having grayscale images
            x = x.expand(-1, 3, -1, -1)
        return self.model(x)


class XLMRoberta(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.roberta = XLMRobertaForSequenceClassification.from_pretrained(
            "FacebookAI/xlm-roberta-large", output_attentions=True
        )
        self.roberta.train()
        self.n_classes = n_classes
        self.linear = nn.Linear(self.roberta.config.hidden_size, self.n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden, _, attn = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        hidden_cls = hidden[:, 0]
        logits = self.linear(hidden_cls)
        return logits, attn


class ModelModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        metric_collection: MetricCollection,
        optimizer_cfg: dict,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer_cfg = optimizer_cfg

        self.save_hyperparameters(ignore=["model", "loss", "metric_collection"])

        self.train_metrics = metric_collection.clone(prefix="train_")
        self.val_metrics = metric_collection.clone(prefix="val_")
        self.test_metrics = metric_collection.clone(prefix="test_")

    def forward(self, x) -> Tensor:
        return self.model(x)

    def _shared_step(self, batch):
        # The `*_` is used to unpack and ignore any additional elements in the batch tuple beyond features and targets.
        # Used for compatibility with external implementation of CIFAR-N datasets where other element represent the index.
        features, targets, *_ = batch
        logits = self(features)
        targets = targets.squeeze().long()
        loss = self.loss(logits, targets)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, targets, predicted_labels

    def training_step(self, batch, batch_idx):
        train_loss, targets, predicted_labels = self._shared_step(batch)
        self.log("train_loss", train_loss)
        self.train_metrics(predicted_labels, targets)
        self.log_dict(self.train_metrics)

        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss, targets, predicted_labels = self._shared_step(batch)
        self.log("val_loss", val_loss)
        self.val_metrics(predicted_labels, targets)
        self.log_dict(self.val_metrics)

    def test_step(self, batch, batch_idx):
        test_loss, targets, predicted_labels = self._shared_step(batch)
        self.log("test_loss", test_loss)
        self.test_metrics(predicted_labels, targets)
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        optim_class = getattr(torch.optim, self.optimizer_cfg["optimizer"])
        return optim_class(self.parameters(), **self.optimizer_cfg["args"])


class RevisionModelModule(ModelModule):
    def __init__(
        self,
        model: nn.Module,
        loss: ReweightCorrectionLoss | Reweighting_Revision_Loss,
        metric_collection: MetricCollection,
        optimizer_cfg: dict,
        # initial_transition_matrix: Tensor,
    ) -> None:
        super().__init__(model, loss, metric_collection, optimizer_cfg)
        self.noise_adaptation = nn.Linear(self.model.n_classes, self.model.n_classes, bias=False)
        # self.noise_adaptation.weight = nn.Parameter(initial_transition_matrix, requires_grad=True)
        self.T_matrix = None
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, revision=False):  # type: ignore
        if revision:
            return self.model(x), self.noise_adaptation.weight
        else:
            return self.model(x)

    def training_step(self, batch, batch_idx):
        # The `*_` is used to unpack and ignore any additional elements in the batch tuple beyond features and targets.
        # Used for compatibility with external implementation of CIFAR-N datasets where other element represent the index.
        features, targets, *_ = batch
        logits, correction = self(features, revision=True)
        targets = targets.squeeze().long()
        train_loss = self.loss(logits, correction, targets)
        self.T_matrix = self.loss.T + correction
        self.T_matrix = self.T_matrix.softmax(dim=1)
        prob_logits = F.softmax(logits, dim=1)
        # Correction step
        out_forward = torch.matmul(self.T_matrix.t(), prob_logits.t()).t()
        predicted_labels = torch.max(out_forward, dim=1)[1]
        ce_train = self.ce(logits, targets)
        self.log("train_ce", ce_train)
        self.log("train_loss", train_loss)
        self.train_metrics(predicted_labels, targets)
        self.log_dict(self.train_metrics)

        return train_loss

    def validation_step(self, batch, batch_idx):
        features, targets, *_ = batch
        logits = self(features, revision=False)
        correction = self.noise_adaptation.weight
        targets = targets.squeeze().long()
        val_loss = self.loss(logits, correction, targets)
        self.T_matrix = self.loss.T + correction
        self.T_matrix = self.T_matrix.softmax(dim=1)
        prob_logits = F.softmax(logits, dim=1)
        out_forward = torch.matmul(self.T_matrix.t(), prob_logits.t()).t()
        predicted_labels = torch.argmax(out_forward, dim=1)
        ce_val = self.ce(logits, targets)
        self.log("val_ce", ce_val)
        self.log("val_loss", val_loss)
        self.val_metrics(predicted_labels, targets)
        self.log_dict(self.val_metrics)

    def test_step(self, batch, batch_idx):
        features, targets, *_ = batch
        logits = self(features, revision=False)
        correction = self.noise_adaptation.weight
        targets = targets.squeeze().long()
        test_loss = self.loss(logits, correction, targets)
        predicted_labels = torch.argmax(logits, dim=1)
        ce_test = self.ce(logits, targets)
        self.log("test_loss", test_loss)
        self.log("test_ce", ce_test)
        self.test_metrics(predicted_labels, targets)
        self.log_dict(self.test_metrics)
