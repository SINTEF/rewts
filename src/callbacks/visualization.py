import os
from typing import Any, Callable, Dict, List, Optional, Union

import darts.models.forecasting.pl_forecasting_module
import darts.utils.data.sequential_dataset
import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import pytorch_lightning.loggers.tensorboard
import torch
from pytorch_lightning.callbacks import Callback as plCallback
from pytorch_lightning.utilities.types import STEP_OUTPUT

import src.utils.plotting
from src import utils

log = utils.get_pylogger(__name__)

_MAX_N_AXIS = 8


class PredictionPlotterCallback(
    plCallback
):  # TODO: support for providing timestamp of predictions to plot?
    """Pytorch Lightning callback to visualize model outputs during training / validation /
    prediction."""

    def __init__(
        self,
        logger: Optional[
            Union[str, pytorch_lightning.loggers.tensorboard.TensorBoardLogger]
        ] = None,  # TODO: some special value that says it should use trainer.logger?
        val_plots_per_epoch: Optional[int] = None,
        train_plots_per_epoch: Optional[int] = None,
        data_names: Optional[Dict[str, str]] = None,
        plot_covariates=True,
    ):  # TODO: change to ish plot_only_these_data (should be possible to get the names from trainer/model already)
        self.logger = logger
        if not isinstance(self.logger, list):
            self.logger = [self.logger]
        self.val_plots_per_epoch = val_plots_per_epoch
        self.train_plots_per_epoch = train_plots_per_epoch
        self._val_log_interval = None
        self._train_log_interval = None
        self.data_names = data_names
        self.plot_covariates = plot_covariates
        super().__init__()

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        assert issubclass(
            type(pl_module), darts.models.forecasting.pl_forecasting_module.PLForecastingModule
        ), "This callback only supports subclasses of PLForecastingModule"
        if getattr(trainer, "loggers", None) is not None:
            self.logger = []
            for logger in trainer.loggers:
                if src.utils.plotting.is_supported_presenter(logger):
                    self.logger.append(logger)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train begins."""
        if self.train_plots_per_epoch is not None:
            self._train_log_interval = max(
                trainer.num_training_batches // (self.train_plots_per_epoch - 1), 1
            )
        if self.data_names is None:
            try:
                self.data_names = {}
                self.data_names["targets"] = (
                    trainer.train_dataloader.dataset.datasets.ds.target_series[
                        0
                    ].components.values.tolist()
                )
                if (
                    getattr(trainer.train_dataloader.dataset.datasets.ds, "covariates", None)
                    is not None
                ):  # TODO: what if it has both future and past?
                    self.data_names[
                        f"{trainer.train_dataloader.dataset.datasets.ds.covariate_type.value}_covariates"
                    ] = trainer.train_dataloader.dataset.datasets.ds.covariates[
                        0
                    ].components.values.tolist()
            except Exception as e:
                log.exception("Could not get data_names")

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Called when the val epoch begins."""
        if self.val_plots_per_epoch is not None:
            self._val_log_interval = [
                max(trainer.num_val_batches[dl_i] // (self.val_plots_per_epoch - 1), 1)
                for dl_i in range(len(trainer.num_val_batches))
            ]

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the validation batch ends."""
        if (
            self._val_log_interval is not None
            and batch_idx % self._val_log_interval[dataloader_idx] == 0
        ):
            item_idx = 0
            fig_name = f"Validation batch {batch_idx}/{trainer.num_val_batches[dataloader_idx]} item {item_idx}"
            fig = self.make_batch_plot_figure(
                trainer, pl_module, outputs, batch, batch_idx, fig_name=fig_name, item_idx=item_idx
            )
            self._present_figure(trainer=trainer, fig=fig, fig_name=fig_name)

    def on_train_batch_end(  # TODO: since training batches are randomized in order, this will plot different samples each epoch
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.
        """
        # TODO: how general is this? Perhaps check on initialization that the model supports this
        if self._train_log_interval is not None and batch_idx % self._train_log_interval == 0:
            item_idx = 0
            fig_name = f"Training batch {batch_idx}/{trainer.num_training_batches} item {item_idx}"
            fig = self.make_batch_plot_figure(
                trainer, pl_module, outputs, batch, batch_idx, fig_name=fig_name, item_idx=item_idx
            )
            self._present_figure(trainer=trainer, fig=fig, fig_name=fig_name)

    def make_batch_plot_figure(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        fig_name: str,
        item_idx: int = 0,
    ) -> plt.Figure:
        """Function to create visualization from a batch of data.

        :param trainer:
        :param pl_module:
        :param outputs:
        :param batch:
        :param batch_idx:
        :param fig_name:
        :param item_idx:
        :return:
        """
        # unpack batch into the different data types
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            future_target,
        ) = (None, None, None, None, None, None)
        if isinstance(
            pl_module, darts.models.forecasting.pl_forecasting_module.PLMixedCovariatesModule
        ):
            (
                past_target,
                past_covariates,
                historic_future_covariates,
                future_covariates,
                static_covariates,
                future_target,
            ) = (b[item_idx] if b is not None else None for b in batch)
        elif isinstance(
            pl_module, darts.models.forecasting.pl_forecasting_module.PLPastCovariatesModule
        ):
            past_target, past_covariates, static_covariates, future_target = (
                b[item_idx] if b is not None else None for b in batch
            )
        elif isinstance(
            pl_module, darts.models.forecasting.pl_forecasting_module.PLDualCovariatesModule
        ):
            (
                past_target,
                historic_future_covariates,
                future_covariates,
                static_covariates,
                future_target,
            ) = (b[item_idx] if b is not None else None for b in batch)
        else:
            raise NotImplementedError("Unsupported dataset type")

        # compute model output for the chosen batch sample
        with torch.no_grad():
            model_output = pl_module._produce_train_output(
                b[item_idx, :, :].unsqueeze(0) if b is not None else None for b in batch[:-1]
            )

            model_std = None
            # For probabilistic models, plot the mean and dispersion of the distribution, i.e. 95% confidence interval
            if model_output.size(-1) > 1:
                assert pl_module.likelihood is not None
                dist_params = pl_module.likelihood._params_from_output(model_output)
                dist = pl_module.likelihood._distr_from_params(dist_params)

                model_output = None
                for center in ["mean", "mode", "median"]:
                    if not torch.all(torch.isnan(getattr(dist, center, torch.nan))):
                        model_output = getattr(dist, center)
                        break
                if (
                    model_output is None
                ):  # not successful in getting a first moment of the distribution.
                    return src.utils.plotting.create_figure(1, 1)[0]

                if not (
                    torch.all(torch.isinf(dist.variance)) or torch.all(torch.isnan(dist.variance))
                ):
                    model_std = torch.sqrt(dist.variance).squeeze(0)
                elif hasattr(dist, "scale") and not (
                    torch.all(torch.isinf(dist.scale)) or torch.all(torch.isnan(dist.scale))
                ):
                    model_std = dist.scale.squeeze(0)
            else:
                model_output = model_output.squeeze(-1)
            model_output = model_output.squeeze(0)
            loss = pl_module.criterion(model_output, future_target)

        n_past_target = past_target.size(-1)

        n_axis = n_past_target
        n_past_covariates = 0
        n_future_covariates = 0
        if self.plot_covariates:  # TODO: control over max number?
            n_past_covariates = past_covariates.size(-1) if past_covariates is not None else 0
            n_future_covariates = (
                future_covariates.size(-1) if future_covariates is not None else 0
            )
            if n_past_covariates > 0 and n_axis + n_past_covariates >= _MAX_N_AXIS:
                past_covariates_separate = False
                n_axis += 1
            else:
                past_covariates_separate = True
                n_axis += n_past_covariates
            if n_future_covariates > 0 and n_axis + n_future_covariates >= _MAX_N_AXIS:
                future_covariates_separate = False
                n_axis += 1
            else:
                future_covariates_separate = True
                n_axis += n_future_covariates

        fig, axs = src.utils.plotting.create_figure(
            nrows=n_axis, ncols=1, sharex=True, figsize=(6.4, 2.4 * n_axis)
        )
        fig.suptitle(fig_name + f", {pl_module.criterion} = {loss:.4f}")
        if isinstance(pl_module, darts.models.forecasting.rnn_model._RNNModule):
            past_time_index = list(range(-1, model_output.shape[0] - 1))
            future_time_index = list(range(model_output.shape[0]))
        else:
            past_time_index = list(range(-pl_module.input_chunk_length, 0))
            future_time_index = list(
                range(
                    -(model_output.size(0) - pl_module.output_chunk_length),
                    pl_module.output_chunk_length,
                )
            )  # TODO: what if output_length > input_length?
        if (
            self.data_names is None
            or self.data_names.get("targets", None) is None
            or len(self.data_names.get("targets", [])) != n_past_target
        ):
            target_names = [str(i) for i in range(n_past_target)]
        else:
            target_names = self.data_names["targets"]
        for feature_i in range(n_past_target):
            self._plot_tensor_line(
                axs[feature_i],
                past_time_index,
                past_target[:, feature_i],
                label="past",
                alpha=0.75,
                linestyle="dashed",
            )
            self._plot_tensor_line(
                axs[feature_i], future_time_index, future_target[:, feature_i], label="target"
            )
            self._plot_tensor_line(
                axs[feature_i],
                future_time_index,
                model_output[:, feature_i],
                label="predicted",
                alpha=0.75,
            )
            if model_std is not None:
                self._plot_tensor_confidence_interval(
                    axs[feature_i],
                    future_time_index,
                    model_output[:, feature_i],
                    model_std[:, feature_i],
                    color=axs[feature_i].lines[-1].get_color(),
                )
            with torch.no_grad():
                feature_i_loss = pl_module.criterion(
                    model_output[:, feature_i], future_target[:, feature_i]
                )
            axs[feature_i].set_title(
                f"Target: {self._format_data_name(target_names[feature_i], max_length=40)}, {pl_module.criterion} = {feature_i_loss:.4f}"
            )
            if feature_i == 0:
                axs[feature_i].legend()

        if self.plot_covariates:
            # TODO: encoders, how to get names?
            if (
                self.data_names is None
                or self.data_names.get("past_covariates", None) is None
                or len(self.data_names["past_covariates"]) != n_past_covariates
            ):
                past_covariate_names = [str(i) for i in range(n_past_covariates)]
            else:
                past_covariate_names = self.data_names["past_covariates"]
                if len(past_covariate_names) < n_past_covariates:
                    past_covariate_names = [str(i) for i in range(n_past_covariates)]

            # TODO: encoders, how to get names?
            if (
                self.data_names is None
                or self.data_names.get("future_covariates", None) is None
                or len(self.data_names["future_covariates"]) != n_future_covariates
            ):
                future_covariate_names = [str(i) for i in range(n_future_covariates)]
            else:
                future_covariate_names = self.data_names["future_covariates"]
                if len(future_covariate_names) < n_future_covariates:
                    future_covariate_names = [str(i) for i in range(n_future_covariates)]

            for feature_i in range(n_past_covariates):
                if past_covariates_separate:
                    self._plot_tensor_line(
                        axs[feature_i + n_past_target],
                        past_time_index,
                        past_covariates[:, feature_i],
                        label="past",
                    )
                    axs[feature_i + n_past_target].set_title(
                        f"Past covariate: {self._format_data_name(past_covariate_names[feature_i], max_length=54)}"
                    )
                else:
                    self._plot_tensor_line(
                        axs[n_past_target],
                        past_time_index,
                        past_covariates[:, feature_i],
                        label=self._format_data_name(
                            past_covariate_names[feature_i], max_length=24
                        ),
                    )

            if n_past_covariates > 0 and not past_covariates_separate:
                axs[n_past_target].set_title("Past covariates")
                axs[n_past_target].legend()

            ax_i_start = n_past_target + (n_past_covariates if past_covariates_separate else 1)
            for feature_i in range(n_future_covariates):
                if future_covariates_separate:
                    self._plot_tensor_line(
                        axs[feature_i + ax_i_start],
                        future_time_index,
                        future_covariates[:, feature_i],
                        label="future",
                    )
                    axs[feature_i + ax_i_start].set_title(
                        f"Future covariate: {self._format_data_name(future_covariate_names[feature_i], max_length=54)}"
                    )
                else:
                    self._plot_tensor_line(
                        axs[ax_i_start],
                        future_time_index,
                        future_covariates[:, feature_i],
                        label=self._format_data_name(
                            future_covariate_names[feature_i], max_length=24
                        ),
                    )

            if n_future_covariates > 0 and not future_covariates_separate:
                axs[ax_i_start].legend()
                axs[ax_i_start].set_title("Future covariates")

        return fig

    def _present_figure(self, trainer, fig, fig_name):
        """Present figure.

        :param trainer:
        :param fig:
        :param fig_name:
        :return:
        """
        for logger in self.logger:
            if isinstance(logger, pytorch_lightning.loggers.tensorboard.TensorBoardLogger):
                src.utils.plotting.present_figure(
                    fig, logger, global_step=trainer.current_epoch, tag=fig_name
                )
            else:
                fig_name = fig_name.replace("/", "-")
                fig_name = f"{fig_name} epoch {trainer.current_epoch}.png"
                if isinstance(logger, pytorch_lightning.loggers.MLFlowLogger):
                    fig_name = os.path.join("prediction_plotter", fig_name)
                else:
                    fig_name = os.path.join(
                        trainer.default_root_dir, "prediction_plotter", fig_name
                    )
                src.utils.plotting.present_figure(fig, logger, fname=fig_name)

    def _format_data_name(self, name: str, max_length: int = 40, characters_per_word: int = 4):
        if len(name) <= max_length:
            return name

        if " " in name:
            sep = " "
        elif "_" in name:
            sep = "_"
        elif "-" in name:
            sep = "-"
        else:
            if len(name) > max_length:
                return name[: max_length - 3] + "..."
            else:
                return name

        name_split = name.split(sep)
        shortened_name = sep.join(
            w[:characters_per_word] if len(w) > characters_per_word else w for w in name_split
        )

        if len(shortened_name) > max_length:
            return shortened_name[: max_length - 3] + "..."
        else:
            return shortened_name

    def _plot_tensor_line(
        self, plotter: Any, x: List, y: torch.Tensor, **plt_kwargs
    ) -> plt.Line2D:
        """Helper function to plot a tensor, which will detach and move the tensor to cpu before
        plotting.

        :param plotter: An object implementing a .plot function, e.g. plt or an axis object.
        :param x: The x data to be plotted
        :param y: The y data to be plotted
        :param plt_kwargs: Additional kwargs sent to the .plot function.
        :return: The plotted line
        """
        return plotter.plot(x, y.detach().cpu(), **plt_kwargs)

    def _plot_tensor_confidence_interval(
        self, plotter: Callable, x: List, y: torch.Tensor, y_std: torch.Tensor, **plt_kwargs
    ) -> matplotlib.collections.PolyCollection:
        """Helper function to plot the dispersion of a stochastic model, given by the mean output
        (y) and the standard deviation of the output (y_std). The function will then plot the 95%
        confidence interval of y (i.e. +- 2 std).

        :param plotter: An object implementing a .fill_between function, e.g. plt or an axis
            object.
        :param x: The x data to be plotted
        :param y: The mean output to be plotted
        :param y_std: The standard deviation of the output
        :param plt_kwargs: Additional kwargs sent to the .plot function.
        :return: The plotted polygons of the confidence interval
        """
        lower_line = y - 2 * y_std
        upper_line = y + 2 * y_std

        if "alpha" not in plt_kwargs:
            plt_kwargs["alpha"] = 0.25

        return plotter.fill_between(
            x, lower_line.detach().cpu(), upper_line.detach().cpu(), **plt_kwargs
        )
