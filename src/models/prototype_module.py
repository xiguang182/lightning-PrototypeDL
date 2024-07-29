from typing import Any, Dict, Tuple

import torchvision
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import MulticlassF1Score
import wandb
from src.models.components.model_helper import list_of_distances, list_of_norms


class ProtoLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        lambdas: Tuple[float, float, float, float],
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.lambdas = lambdas

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # metric object for calculating f1 score
        self.train_f1 = MulticlassF1Score(num_classes=10, multidim_average= 'global', average=None)
        self.val_f1 = MulticlassF1Score(num_classes=10, multidim_average= 'global', average=None)
        self.test_f1 = MulticlassF1Score(num_classes=10, multidim_average= 'global', average=None)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.train_f1.reset()
        self.val_f1.reset()
        self.test_f1.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        ce_loss = self.criterion(logits, y)
        
        prototypes = self.net.prototype_layer.prototypes
        feature_vectors = self.net.feature_vectors
        lambda_1_loss = torch.mean(torch.min(list_of_distances(prototypes, feature_vectors.view(-1, self.net.in_channels_prototype)), dim=1)[0])
        lambda_2_loss = torch.mean(torch.min(list_of_distances(feature_vectors.view(-1, self.net.in_channels_prototype), prototypes), dim=1)[0])
        
        out_decoder = self.net.decoder(feature_vectors)
        # autoencoder loss, criterion is the mean squared error. This loss not modulized.
        ae_loss = torch.mean(list_of_norms(out_decoder - x))
        # how to pass lambdas? or just hard code them here? -- in lightning module config file
        loss = [ce_loss, lambda_1_loss, lambda_2_loss, ae_loss]
        
        # sum the losses
        loss = sum([a * b for a, b in zip(loss, self.lambdas)])

        preds = torch.argmax(logits, dim=1)
        return loss, preds, y, out_decoder

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, decoded_imgs = self.model_step(batch)
        # print(decoded_imgs.shape) torch.Size([128, 1, 28, 28])
        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1[0], on_step=False, on_epoch=True, prog_bar=True,  metric_attribute="train/f1/class_0")
        
        # hard coded for wandb logging due to wandb.Image object conversion
        if batch_idx % 100 == 0:            
            input_imgs = batch[0][:8]
            decoded_imgs = decoded_imgs[:8]
            imgs = torch.cat([input_imgs, decoded_imgs], dim=0)
            grid = torchvision.utils.make_grid(imgs)
            # print(grid.shape) make grid returns a tensor of shape (3, 64, 242) duplicate the first channel to make it (3, 64, 242)
            self.logger.experiment.log(
                {"train images (first row) and decoded images (second row)": wandb.Image(grid[0], caption=f"Epoch {self.current_epoch}, batch {batch_idx}00,  Loss {loss}")}
            )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, _ = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        # if batch_idx % 100 == 0:
        #     print(batch[0].shape)
            
        #     imgs = batch[0][:8]
        #     grid = torchvision.utils.make_grid(imgs)
        #     self.logger.experiment.log(
        #         {"samples": [wandb.Image(img, caption="batch ${idx}00") for idx, img in enumerate(grid)]}
        #     )
        self.val_f1(preds, targets)
        self.log("val/f1", self.val_f1[0], on_step=False, on_epoch=True, prog_bar=True, metric_attribute="val/f1/class_0")

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        # print(f"val/acc: {acc}")
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

        f1 = self.val_f1.compute()
        # print(f"val/f1: {f1}")
        self.val_f1_best(f1[0])
        tmp = self.val_f1_best.compute()
        # print(f"val/f1_best: {tmp}")
        self.log("val/f1_best", tmp, sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, _ = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.test_f1(preds, targets)
        self.log("test/f1", self.test_f1[0], on_step=False, on_epoch=True, prog_bar=True, metric_attribute="test/f1/class_0")

        # log images todo
        # if batch_idx % 100 == 0:
        #     batch = batch[:8]
        #     grid = torchvision.utils.make_grid(x)
        #     self.logger.experiment.add_image("test/input", grid, self.global_step)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    
    _ = ProtoLitModule(None, None, None, None, None)
