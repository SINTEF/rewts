import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
from darts.models.forecasting.pl_forecasting_module import io_processor
from darts.models.forecasting.rnn_model import CustomRNNModule


class NoTargetRNNModule(CustomRNNModule):
    """This class allows to create custom RNN modules that can later be used with Darts'
    `RNNModel`. It adds the backbone that is required to be used with Darts'
    `TorchForecastingModel` and `RNNModel`.

    To create a new module, subclass from `CustomRNNModule` and:

    * Define the architecture in the module constructor (`__init__()`)

    * Add the `forward()` method and define the logic of your module's forward pass

    * Use the custom module class when creating a new `RNNModel` with parameter `model`.

    You can use `darts.models.forecasting.rnn_model._RNNModule` as an example.

    Parameters
    ----------
    input_size
        The dimensionality of the input time series.
    hidden_dim
        The number of features in the hidden state `h` of the RNN module.
    num_layers
        The number of recurrent layers.
    target_size
        The dimensionality of the output time series.
    nr_params
        The number of parameters of the likelihood (or 1 if no likelihood is used).
    dropout
        The fraction of neurons that are dropped in all-but-last RNN layers.
    **kwargs
        all parameters required for `darts.models.forecasting.pl_forecasting_module.PLForecastingModule`
        base class.
    """

    def __init__(
        self,
        input_size: int,
        name: str = "LSTM",
        **kwargs,
    ):
        assert (
            kwargs["train_sample_shape"][1] is not None
        ), "NoTargetRNNModule requires future_covariates"
        # size of input is only future_covariates, not target as well
        input_size = kwargs["train_sample_shape"][1][1]

        # RNNModule doesn't really need input and output_chunk_length for PLModule
        super().__init__(input_size=input_size, **kwargs)
        self.name = name

        # Defining the RNN module
        self.rnn = getattr(nn, name)(
            self.input_size,
            self.hidden_dim,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        # The RNN module needs a linear layer V that transforms hidden states into outputs, individually
        self.V = nn.Linear(self.hidden_dim, self.target_size * self.nr_params)

    @io_processor
    def forward(
        self, x_in: Tuple, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of RNN."""
        x, _ = x_in
        # data is of size (batch_size, input_length, input_size)
        batch_size = x.shape[0]

        # out is of size (batch_size, input_length, hidden_dim)
        out, last_hidden_state = self.rnn(x) if h is None else self.rnn(x, h)

        # Here, we apply the V matrix to every hidden state to produce the outputs
        predictions = self.V(out)

        # predictions is of size (batch_size, input_length, target_size)
        predictions = predictions.view(batch_size, -1, self.target_size, self.nr_params)

        # returns outputs for all inputs, only the last one is needed for prediction time
        return predictions, last_hidden_state

    def _produce_train_output(self, input_batch: Tuple) -> torch.Tensor:
        """Take standard input to model, and extract the relevant time series.

        In this case, we discard the target input time series.
        """
        (
            past_target,
            historic_future_covariates,
            future_covariates,
            static_covariates,
        ) = input_batch
        model_input = (
            future_covariates,
            static_covariates,
        )
        return self(model_input)[0]

    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> torch.Tensor:
        """This model is recurrent, so we have to write a specific way to obtain the time series
        forecasts of length n."""
        (
            past_target,
            historic_future_covariates,
            future_covariates,
            static_covariates,
        ) = input_batch

        if historic_future_covariates is None:
            raise ValueError("NoTargetRNN needs future_covariates")

        # RNNs need as inputs (target[t] and covariates[t+1]) so here we shift the covariates
        all_covariates = torch.cat(
            [historic_future_covariates[:, 1:, :], future_covariates], dim=1
        )
        cov_past, cov_future = (
            all_covariates[:, : past_target.shape[1], :],
            all_covariates[:, past_target.shape[1] :, :],
        )
        input_series = cov_past

        batch_prediction = []
        out, last_hidden_state = self._produce_predict_output((input_series, static_covariates))
        batch_prediction.append(out[:, -1:, :])
        prediction_length = 1

        while prediction_length < n:

            # create new input to model from current covariates
            new_input = cov_future[:, prediction_length - 1 : prediction_length, :]

            # feed new input to model, including the last hidden state from the previous iteration
            out, last_hidden_state = self._produce_predict_output(
                (new_input, static_covariates), last_hidden_state
            )

            # append prediction to batch prediction array, increase counter
            batch_prediction.append(out[:, -1:, :])
            prediction_length += 1

        # bring predictions into desired format and drop unnecessary values
        batch_prediction = torch.cat(batch_prediction, dim=1)
        batch_prediction = batch_prediction[:, :n, :]
        return batch_prediction


setattr(sys.modules[CustomRNNModule.__module__], "NoTargetRNNModule", NoTargetRNNModule)
