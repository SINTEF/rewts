from typing import Union, Sequence, Optional, Tuple, List, Callable, Dict, Literal

import darts.models.forecasting.ensemble_model
import darts.utils
import cvxopt
import numpy as np
from darts.timeseries import TimeSeries
from darts.logging import raise_if
from src.utils import pylogger
import pandas as pd
import copy

cvxopt.solvers.options["show_progress"] = False
log = pylogger.get_pylogger(__name__)


class TSEnsembleModel(darts.models.forecasting.ensemble_model.EnsembleModel):
    def __init__(self,
                 models,
                 fit_forecast_horizon=30,
                 fit_stride=1,
                 fit_weights_every=30 * 10,
                 lookback_data_length=720,
                 data_pipelines=None,
                 datamodule=None,
                 autoregressive_mix: bool = True,
                 weight_threshold=1e-4,
                 qp_magnitude_threshold=(1e-5, 1e5),
                 ):
        if data_pipelines is not None:
            assert len(models) == len(data_pipelines), "A data_pipeline per model must be provided."
            assert datamodule is not None, "To normalize data with data_pipelines, a datamodule containing the same data components must be passed."
        assert all([m._fit_called for m in models]), "This ensemble model only works with fitted models"

        self.weights = None
        self._weights_history = None
        self.data_pipelines = data_pipelines
        self.datamodule = datamodule
        self.autoregressive_mix = autoregressive_mix
        self.fit_forecast_horizon = fit_forecast_horizon
        self.fit_stride = fit_stride
        self.fit_weights_every = fit_weights_every
        self.lookback_data_length = lookback_data_length
        self._fit_data = None
        self._fit_weights = True
        self._weight_threshold = weight_threshold
        self._qp_magnitude_threshold = qp_magnitude_threshold
        self._weights_last_update = 0

        super().__init__(models, train_num_samples=1, train_samples_reduction="mean", train_forecasting_models=False)
        self._fit_called = True  # required by darts
        self.reset()

    @property
    def _models_are_probabilistic(self):
        return False

    def reset(self):
        self.weights = np.full((len(self.forecasting_models),), fill_value=1 / len(self.forecasting_models))
        self._weights_history = []  # TODO: some functionality to reset this list

    def save_weights(self, path):
        np.save(path, np.stack(self._weights_history))

    def ensemble(self, predictions, series=None, num_samples: int = 1, predict_likelihood_parameters: bool = False): # TODO: what about multivariate time series?
        if num_samples != 1 or predict_likelihood_parameters:  # Probabilistic Ensemble
            raise NotImplementedError
        return TimeSeries.from_times_and_values(predictions.time_index, predictions.values() @ self.weights.astype(predictions.dtype))

    def fit(
            self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            verbose: bool = True,
    ):
        log.info("Fitting ensemble weights")
        super().fit(series, past_covariates=past_covariates, future_covariates=future_covariates)

        input_chunk_length = abs(self.extreme_lags[0])

        # spare train_n_points points to serve as regression target
        is_single_series = isinstance(series, TimeSeries)
        if is_single_series:
            fit_data_too_short = len(self.training_series) <= input_chunk_length
        else:
            return NotImplementedError("Support for multiple series is not implemented")
            fit_data_too_short = any([len(s) <= input_chunk_length for s in series])

        raise_if(
            fit_data_too_short,
            "fit series is too short (must be at least 1 longer than input_chunk_length of models)",
            log,
        )

        original_fit_weights = self._fit_weights
        self._fit_weights = False

        fit_series = series
        if self._fit_data is not None:
            # TODO: generalize to not just time-shifted data
            forecastable_index = darts.utils.historical_forecasts.utils._get_historical_forecastable_time_index(
                self,
                fit_series,
                past_covariates,
                future_covariates,
                is_training=False,
                reduce_to_bounds=False,  # only returns the (start, end) indices
            )
            fit_series_start = None
            pred_starts = [pred.time_index[0] for pred in self._fit_data[0]]
            # Note that a prediction starts at the next point in time after the prediction point.
            # and that forecastable_index will include one more point than what is forecastable (dont know why)
            # We check if the corresponding prediction for each forecastable point is already in the predictions
            # (this will be the next point in time) thus we check pred_i + 1.
            for pred_i in range(0, len(forecastable_index) - (self.fit_forecast_horizon + 1), self.fit_stride):
                if forecastable_index[pred_i + 1] not in pred_starts:
                    fit_series_start = series.get_index_at_point(forecastable_index[pred_i + 1]) - input_chunk_length
                    fit_series = series[fit_series_start:]
                    break

            if fit_series_start is None:  # no new data
                return self

        pred_data = {"series": fit_series, "past_covariates": past_covariates, "future_covariates": future_covariates}
        predictions = []

        if verbose:
            model_iterator = darts.utils._build_tqdm_iterator(self.forecasting_models, verbose, desc="Computing model predictions on lookback data", total=len(self.forecasting_models))
        else:
            model_iterator = self.forecasting_models

        for m_i, model in enumerate(model_iterator):
            # TODO: will perhaps give different amount of predictions if unequal input_chunk_length?
            # therefore perhaps adjust fit_series per model based on chunk length?
            m_predictions = model.historical_forecasts(
                num_samples=1,
                forecast_horizon=self.fit_forecast_horizon,
                stride=self.fit_stride,
                retrain=False,
                last_points_only=False,
                verbose=False,
                **self._transform_data(m_i, pred_data),
                enable_optimization=False,  # optimization seems to cause issues under some conditions
            )
            predictions.append(self._inverse_transform_data(m_i, m_predictions))

        if self._fit_data is not None:
            # Because we assume only time shift into the future, all new predictions are relevant and come sequentially after the last old relevant prediction
            # Can precompute which predictions are still relevant from before
            for m_i in range(len(self.forecasting_models)):
                predictions[m_i] = [pred for pred in self._fit_data[m_i] if pred.time_index[0] >= forecastable_index[0]] + predictions[m_i]
        self._fit_weights = original_fit_weights
        self._fit_data = predictions

        if self.fit_forecast_horizon == self.fit_stride == 1:
            # TODO: multivariate data (where components > 1)
            predictions = darts.timeseries.concatenate([darts.timeseries.concatenate(m_p) for m_p in predictions], axis=1)

            target = self.training_series.slice_intersect(predictions)

            predictions = predictions.values()
            target = target.values()

            square_weights = (predictions.T @ predictions).astype(np.float64) / predictions.shape[0]
            linear_weights = -(predictions.T @ target).astype(np.float64) / predictions.shape[0]
        else:
            # TODO: generalize for multivariate target (must think where this dimension fits in)
            # TODO: if use fit_series here in place of training_series. If fit_series is a sequence we have to find out
            # first what series in the sequence that the prediction overlaps with before we slice_intersect.
            targets = np.stack([series.slice_intersect(prediction).values() for prediction in predictions[0]])
            predictions = [[ts.values() for ts in inner_list] for inner_list in predictions]
            predictions = np.transpose(np.array(predictions).squeeze(-1), (1, 2, 0))  # Here we squeeze away the n_targets dimensions

            square_weights = np.mean(np.transpose(predictions, axes=(0, 2, 1)) @ predictions, axis=0, dtype=np.float64)
            linear_weights = -np.mean(np.transpose(predictions, axes=(0, 2, 1)) @ targets, axis=0, dtype=np.float64)

        # condition Q and p to avoid numerical issues
        low_threshold, high_threshold = self._qp_magnitude_threshold

        max_val = max(np.max(np.abs(square_weights)), np.max(np.abs(linear_weights)))
        if max_val > high_threshold:
            scale_factor = max_val / high_threshold
        elif max_val < low_threshold and max_val != 0:
            scale_factor = low_threshold / max_val
        else:
            scale_factor = 1  # No scaling needed

        square_weights /= scale_factor
        linear_weights /= scale_factor

        # constraints

        # each model weight >= 0
        c_models_geq0_rhs = np.zeros((len(self.forecasting_models), 1))
        c_models_geq0_lhs = np.diag(np.ones(len(self.forecasting_models)) * -1)

        # sum of all weights = 1
        c_sum_eq1_rhs = np.ones((1, 1))
        c_sum_eq1_lhs = np.ones((1, len(self.forecasting_models)))

        # Get the weights for the best combination of the models:
        weights = cvxopt.solvers.qp(*map(cvxopt.matrix, [square_weights,
                                                         linear_weights,
                                                         c_models_geq0_lhs,
                                                         c_models_geq0_rhs,
                                                         c_sum_eq1_lhs,
                                                         c_sum_eq1_rhs]))['x']

        self.weights = np.array(weights)
        if self._weight_threshold is not None:
            self.weights[self.weights < self._weight_threshold] = 0
            self.weights = np.true_divide(self.weights, np.sum(self.weights))

        self._weights_history.append(self.weights)

        return self

    def historical_forecasts(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        train_length: Optional[int] = None,
        start: Optional[Union[pd.Timestamp, float, int]] = None,
        start_format: Literal["position", "value"] = "value",
        forecast_horizon: int = 1,
        stride: int = 1,
        retrain: Union[bool, int, Callable[..., bool]] = True,
        overlap_end: bool = False,
        last_points_only: bool = True,
        verbose: bool = False,
        show_warnings: bool = True,
        predict_likelihood_parameters: bool = False,
        enable_optimization: bool = True,
        fit_weights: Optional[bool] = None
    ) -> Union[
        TimeSeries, List[TimeSeries], Sequence[TimeSeries], Sequence[List[TimeSeries]]
    ]:
        if fit_weights is not None:
            self._fit_weights = fit_weights

        if self._fit_weights:
            self._weights_last_update = -self.fit_weights_every  # TODO: or first forecastable?

        return super().historical_forecasts(
                        series=series,
                        past_covariates=past_covariates,
                        future_covariates=future_covariates,
                        num_samples=num_samples,
                        train_length=train_length,
                        start=start,
                        start_format=start_format,
                        forecast_horizon=forecast_horizon,
                        stride=stride,
                        retrain=retrain,
                        overlap_end=overlap_end,
                        last_points_only=last_points_only,
                        verbose=verbose,
                        show_warnings=show_warnings,
                        predict_likelihood_parameters=predict_likelihood_parameters,
                        enable_optimization=enable_optimization,
                    )


    def _split_multi_ts_sequence(
        self, n: int, ts_sequence: Sequence[TimeSeries]
    ) -> Tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:
        left = [ts[:-n] for ts in ts_sequence]
        right = [ts[-n:] for ts in ts_sequence]
        return left, right

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        verbose: bool = False,
        predict_likelihood_parameters: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        # for single-level ensemble, probabilistic forecast is obtained directly from forecasting models
        if self.train_samples_reduction is None:
            pred_num_samples = num_samples
            forecast_models_pred_likelihood_params = predict_likelihood_parameters
        # for multi-levels ensemble, forecasting models can generate arbitrary number of samples
        else:
            pred_num_samples = self.train_num_samples
            # second layer model (regression) cannot be trained on likelihood parameters
            forecast_models_pred_likelihood_params = False

        self._verify_past_future_covariates(past_covariates, future_covariates)

        if self._fit_weights and len(series) >= self.lookback_data_length and len(series) - self._weights_last_update >= self.fit_weights_every:
            self._fit_weights = False
            self.fit(series[-self.lookback_data_length:], past_covariates, future_covariates, verbose=verbose)
            self._fit_weights = True
            self._weights_last_update = len(series)

        if self.autoregressive_mix:
            is_single_series = isinstance(series, TimeSeries) or series is None
            if not is_single_series:
                return NotImplementedError
            predict_series = copy.deepcopy(series)
            model_datas = {}
            for m_i in range(len(self.forecasting_models)):
                m_data = self._transform_data(m_i, dict(
                    series=None,
                    future_covariates=future_covariates,
                    past_covariates=past_covariates
                ))
                for series_type, series_data in m_data.items():
                    if series_type not in model_datas:
                        model_datas[series_type] = []
                    model_datas[series_type].append(series_data)
            for n_i in range(n):
                for m_i in range(len(self.forecasting_models)):
                    model_datas["series"][m_i] = self._transform_data(m_i, dict(series=predict_series))["series"]
                model_predictions = self._make_multiple_predictions(
                    n=1,
                    **model_datas,
                    num_samples=pred_num_samples,
                    predict_likelihood_parameters=forecast_models_pred_likelihood_params,
                    transform_data=False
                )
                prediction = self.ensemble(
                    model_predictions,
                    series=series,
                    num_samples=num_samples,
                    predict_likelihood_parameters=predict_likelihood_parameters,
                )
                predict_series = predict_series.append(prediction)

            return predict_series[-n:]
        else:
            predictions = self._make_multiple_predictions(
                n=n,
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                num_samples=pred_num_samples,
                predict_likelihood_parameters=forecast_models_pred_likelihood_params,
            )

            return self.ensemble(
                predictions,
                series=series,
                num_samples=num_samples,
                predict_likelihood_parameters=predict_likelihood_parameters,
            )

    def _make_multiple_predictions(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        predict_likelihood_parameters: bool = False,
        transform_data: bool = True
    ):
        if num_samples != 1 or predict_likelihood_parameters:    # Probabilistic Ensemble
            raise NotImplementedError

        is_single_series = isinstance(series, TimeSeries) or series is None
        if not is_single_series:
            assert not transform_data, "The only valid use of sequence of data is for pretransformed data"
            assert len(series) == len(self.forecasting_models), "Forecasting a sequence of series is not supported"

        predictions = []
        data = {"series": series, "past_covariates": past_covariates, "future_covariates": future_covariates}

        # TODO: implement joblib parallelization?
        for m_i, model in enumerate(self.forecasting_models):
            if self.weights[m_i] == 0:
                if not is_single_series:
                    m_series = series[m_i]
                else:
                    m_series = series
                predictions.append(darts.utils.timeseries_generation.constant_timeseries(value=0, start=m_series.end_time() + m_series.freq, length=n, freq=m_series.freq, dtype=m_series.dtype))  # TODO: does this work for multivariate?
                continue
            # TODO: check if model supports data
            if transform_data:
                model_data = self._transform_data(m_i, data)
            else:
                model_data = {k: v[m_i] for k, v in data.items()}
            m_prediction = model._predict_wrapper(
                                n=n,
                                num_samples=num_samples,
                                **model_data
                            )
            predictions.append(self._inverse_transform_data(m_i, m_prediction))

        return (
            self._stack_ts_seq(predictions)
            #if is_single_series
            #else self._stack_ts_multiseq(predictions)
        )

    def _transform_data(self, model_index: int, data: Dict[str, TimeSeries]):
        assert "series" in data
        # TODO: could stack into one series, but series / covariates can be different lengths which complicates things
        if self.data_pipelines is None or model_index >= len(self.data_pipelines):
            return data

        res = {}
        for data_name, series in data.items():
            if series is not None:
                # TODO: how to transform part of the data?
                # Stack data?
                res[data_name] = self.datamodule.transform_data(series, pipeline=self.data_pipelines[model_index])
            else:
                res[data_name] = None

        return res

    def _inverse_transform_data(self, model_index: int, series: TimeSeries):  # TODO: support for inverse covariates?
        if self.data_pipelines is not None:
            series = self.datamodule.inverse_transform_data(series, pipeline=self.data_pipelines[model_index], partial=True)
        return series


