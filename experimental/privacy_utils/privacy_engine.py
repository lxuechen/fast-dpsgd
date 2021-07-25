import logging
import math
import types
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn

from experimental.privacy_utils import autograd_grad_sample
from experimental.privacy_utils.accounting import gdp_accounting, rdp_accounting

DEFAULT_ALPHAS = tuple(1 + x / 10.0 for x in range(1, 100)) + tuple(range(12, 64))


class EfficientPrivacyEngine(object):
    """Differentially-private SGD engine.

    >>> model = nn.Linear(10, 10)
    >>> pe = PrivacyEngine(module=model)

    Args:
        module: The PyTorch module for which per-sample gradient is required.
            Setting the `requires_grad` attribute of a parameter to False
            disables the per-sample gradient accumulation.
        batch_size: The expected lot size.
        max_grad_norm: The maximum 2-norm for gradient clipping.
        epochs: The number of epochs for training.
        noise_multiplier: The extra multiplier for DP-SGD noise.
        target_epsilon: The target privacy spending. Only used to estimate the
            `noise_multiplier` if it is not set.
        target_delta: The target failure probability. Defaults to 1 / (2 * sample_size) if None.
        gradient_accumulation_steps: The number of gradient accumulation steps.
        loss_reduction: One of (`mean`, 'sum').
            Should correspond to how the loss is scaled; if set correctly,
            two cases should give the same aggregated per-sample gradient.
        accounting_mode: One of (`rdp`, `gdp`, `all`, `rdp_cks`). Privacy accounting mode.
        alphas: The RDP orders for (ε, δ)-DP conversion.

    Notes:
        When using virtual batches, make sure to divide the per-chunk loss by
        the total number of chunks. This is mostly to make consistent with
        Huggingface's `trainer.py`. The engine's `step` compensates for this
        loss at the following line

        @formatter:off
        https://github.com/lxuechen/private_nlp/blob/6fbc678c0cb9472246d197e6f6c49d556c834629/privacy_utils/privacy_engine.py#L243
        @formatter:on
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        batch_size: int,
        sample_size: int,
        max_grad_norm: float,
        epochs: Optional[int] = None,
        noise_multiplier: Optional[float] = None,
        target_epsilon: Optional[float] = None,
        target_delta: Optional[float] = None,
        gradient_accumulation_steps: int = 1,
        loss_reduction="mean",
        batch_first: bool = True,
        accounting_mode="rdp_cks",
        alphas: Sequence[float] = DEFAULT_ALPHAS,
        verbose: bool = False,
        named_params: Optional[Sequence] = None,
        fp16=False,
        **_,
    ):
        super(EfficientPrivacyEngine, self).__init__()
        if accounting_mode not in ('rdp', 'gdp', 'all', 'rdp_cks'):
            raise ValueError(f"Unknown accounting mode: {accounting_mode}")

        # Privacy parameters.
        sample_rate = batch_size / sample_size
        if target_delta is None:
            target_delta = 1 / 2 / sample_size
        if noise_multiplier is None:
            if target_epsilon is None or epochs is None:
                raise ValueError(
                    f"`target_epsilon` and `epochs` must be specified when `noise_multiplier` is `None`."
                )
            if accounting_mode == "rdp":
                noise_multiplier = get_sigma_from_rdp(
                    target_epsilon=target_epsilon,
                    target_delta=target_delta,
                    sample_rate=sample_rate,
                    epochs=epochs,
                    alphas=alphas,
                )
            elif accounting_mode == "rdp_cks":
                noise_multiplier = get_sigma_from_rdp_cks(
                    target_epsilon=target_epsilon,
                    target_delta=target_delta,
                    sample_rate=sample_rate,
                    epochs=epochs,
                    alphas=alphas,
                )
            else:
                noise_multiplier = get_sigma_from_gdp(
                    target_epsilon=target_epsilon,
                    target_delta=target_delta,
                    sample_rate=sample_rate,
                    epochs=epochs,
                )

        self.batch_size = batch_size
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.max_grad_norm = max_grad_norm

        self.epochs = epochs
        self.noise_multiplier = noise_multiplier
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.loss_reduction = loss_reduction
        self.alphas = alphas
        self.accounting_mode = accounting_mode
        self.batch_first = batch_first
        self.verbose = verbose
        self.fp16 = fp16

        # Internals.
        self.steps = 0  # Tracks privacy spending.

        # Recording.
        self.max_clip = None
        self.min_clip = None
        self.med_clip = None
        self.signal = None
        self.noise = None
        self.snr = None
        self.noise_limit = None

        self.module = module
        if named_params is None:
            self.named_params = tuple(
                (name, param) for (name, param) in module.named_parameters() if param.requires_grad
            )
        else:
            self.named_params = named_params
        self.num_params = sum(param.numel() for _, param in self.named_params)

    def attach(self, optimizer):
        autograd_grad_sample.add_hooks(
            self.module, batch_first=self.batch_first, loss_reduction=self.loss_reduction
        )

        # Override zero grad.
        def dp_zero_grad(_self):
            _self.privacy_engine.zero_grad()

        # Override step.
        def dp_step(_self, closure=None):
            _self.privacy_engine.step()
            _self.original_step(closure)

        def virtual_step(_self):
            _self.privacy_engine.virtual_step()

        optimizer.privacy_engine = self

        optimizer.original_step = optimizer.step
        optimizer.step = types.MethodType(dp_step, optimizer)

        optimizer.original_zero_grad = optimizer.zero_grad
        optimizer.zero_grad = types.MethodType(dp_zero_grad, optimizer)

        optimizer.virtual_step = types.MethodType(virtual_step, optimizer)

        # Just to be safe, we also override `zero_grad` for module.
        self.module.privacy_engine = self
        self.module.original_zero_grad = self.module.zero_grad
        self.module.zero_grad = types.MethodType(dp_zero_grad, self.module)

    @torch.no_grad()
    def step(self):
        self.steps += 1
        if not self.fp16:
            self.create_noisy_grads()

    def create_noisy_grads(self, extra_factor=1.):
        # Add noise and scale by inverse batch size.
        signals, noises = [], []
        for name, param in self.named_params:
            # This is only True when there are previous virtual steps.
            # The .grad contains the summed clipped gradients of this batch.
            # Summed clipped gradients of previous batches are in .summed_grad.
            # When there's no gradient accumulation, .summed_grad is not created.
            if hasattr(param, 'summed_grad'):
                param.grad += param.summed_grad
            signals.append(param.grad.reshape(-1).norm(2))

            if self.noise_multiplier > 0 and self.max_grad_norm > 0:
                noise = torch.normal(
                    mean=0,
                    std=self.noise_multiplier * self.max_grad_norm * extra_factor,  # Very important in fp16!
                    size=param.size(),
                    device=param.device,
                    dtype=param.dtype,
                )
                param.grad += noise
                noises.append(noise.reshape(-1).norm(2))
                del noise
            if self.loss_reduction == "mean":
                param.grad /= self.batch_size

        if noises:
            self.signal, self.noise = tuple(torch.stack(lst).norm(2).item() for lst in (signals, noises))
            self.noise_limit = math.sqrt(self.num_params) * self.noise_multiplier * self.max_grad_norm
            self.snr = self.signal / self.noise
            if self.verbose:
                logging.warning(
                    f"signal: {self.signal:6f}, "
                    f"noise: {self.noise:.6f}, "
                    f"snr: {self.snr:.6f}, "
                    f"sqrt(d) C sigma: {self.noise_limit:.4f}, \n"
                    f"num_params: {self.num_params}"
                )
        else:
            self.snr = math.inf  # Undefined!

    def zero_grad(self):
        for name, param in self.named_params:
            if hasattr(param, "grad"):
                del param.grad
            if hasattr(param, "norm_sample"):
                del param.norm_sample
            if hasattr(param, "summed_grad"):
                del param.summed_grad
            if hasattr(param, "grad_sample"):
                del param.grad_sample

    def virtual_step(self):
        for name, param in self.named_params:
            if hasattr(param, 'summed_grad'):
                param.summed_grad += param.grad
            else:
                param.summed_grad = param.grad

            if hasattr(param, "grad"):
                del param.grad
            if hasattr(param, "norm_sample"):
                del param.norm_sample
            if hasattr(param, "grad_sample"):
                del param.grad_sample

    def get_norm_sample(self):
        norm_sample = torch.stack([param.norm_sample for name, param in self.named_params], dim=0).norm(2, dim=0)
        return norm_sample

    def get_coef_sample(self):
        norm_sample = self.get_norm_sample()
        return torch.clamp_max(self.max_grad_norm / (norm_sample + 1e-6), 1.)

    def get_privacy_spent(self, steps=None) -> Dict:
        if steps is None:
            steps = self.steps

        privacy_results = {}

        kwargs = dict(
            sample_rate=self.sample_rate,
            steps=steps,
            delta=self.target_delta,
            sigma=self.noise_multiplier,
            alphas=self.alphas,
        )
        if self.accounting_mode in ('all', 'rdp'):
            # Uses conversion.
            eps_rdp, alpha_rdp = _eps_from_rdp(**kwargs)
            privacy_results['eps_rdp_opacus'] = eps_rdp
            privacy_results['alpha_rdp_opacus'] = alpha_rdp

        if self.accounting_mode in ('all', 'gdp'):
            eps_gdp, mu_gdp = _eps_from_gdp(**kwargs)
            privacy_results['eps_gdp'] = eps_gdp
            privacy_results['mu_gdp'] = mu_gdp

        if self.accounting_mode in ('all', "rdp_cks"):
            eps_rdp, alpha_rdp = _eps_from_rdp_cks(**kwargs)
            privacy_results['eps_rdp'] = eps_rdp
            privacy_results['alpha_rdp'] = alpha_rdp

        return privacy_results

    def get_privacy_stats(self):
        """Get the clipping, signal, and noise status."""
        return {
            "med_clip": self.med_clip,
            "max_clip": self.max_clip,
            "min_clip": self.min_clip,
            "snr": self.snr,
            "signal": self.signal,
            "noise": self.noise,
            "noise_limit": self.noise_limit,
        }

    def __repr__(self):
        return (
            f"PrivacyEngine(\n"
            f"  target_epsilon={self.target_epsilon}, \n"
            f"  target_delta={self.target_delta}, \n"
            f"  noise_multiplier={self.noise_multiplier}, \n"
            f"  epochs={self.epochs}, \n"
            f"  max_grad_norm={self.max_grad_norm}, \n"
            f"  sample_rate={self.sample_rate}, \n "
            f"  (actual) batch_size={self.batch_size}, \n"
            f"  gradient_accumulation_steps={self.gradient_accumulation_steps}, \n"
            f"  loss_reduction={self.loss_reduction}, \n"
            f"  accounting_mode={self.accounting_mode}, \n"
            f")"
        )


def get_sigma_from_rdp(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: Union[float, int],
    alphas=DEFAULT_ALPHAS,
    threshold=1e-7,
    sigma_hi_init=1e1,
    sigma_lo_init=1e-6,
) -> float:
    """Get noise multiplier σ for a given ε from Renyi-DP accounting.

    Notes:
        Setting `threshold` to an appropriate value is crucial for accurate search.
        The default is fine-grained enough for ε ∈ [0.1, 1e10].

    Args:
        target_epsilon: ε in (ε, δ)-DP.
        target_delta: δ in (ε, δ)-DP.
        sample_rate: Rate for Poisson subsampling, typically denoted as q.
        epochs: Number of passes through the dataset.
        alphas: Orders for Renyi-divergence.
        threshold: Threshold for binary search. Determines the granularity of
            the search result.
        sigma_hi_init: Starting point for the high end of binary search.
        sigma_lo_init: Starting point for the low end of binary search.

    Returns:
        The noise multiplier σ for DP-SGD.
    """
    steps = epochs / sample_rate

    def sigma_to_eps(sigma):
        """Compute ε for a given σ based on Renyi-DP."""
        eps, _ = _eps_from_rdp(
            sample_rate=sample_rate,
            sigma=sigma,
            steps=steps,
            alphas=alphas,
            delta=target_delta,
        )
        return eps

    return _get_sigma_with_target_epsilon(
        sigma_hi_init=sigma_hi_init,
        sigma_lo_init=sigma_lo_init,
        sigma_to_eps=sigma_to_eps,
        target_epsilon=target_epsilon,
        threshold=threshold,
    )


def get_sigma_from_rdp_cks(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: Union[float, int],
    alphas=DEFAULT_ALPHAS,
    threshold=1e-7,
    sigma_hi_init=1e1,
    sigma_lo_init=1e-6,
) -> float:
    steps = epochs / sample_rate

    def sigma_to_eps(sigma):
        """Compute ε for a given σ based on Renyi-DP."""
        eps, _ = _eps_from_rdp_cks(
            sample_rate=sample_rate,
            sigma=sigma,
            steps=steps,
            alphas=alphas,
            delta=target_delta,
        )
        return eps

    return _get_sigma_with_target_epsilon(
        sigma_hi_init=sigma_hi_init,
        sigma_lo_init=sigma_lo_init,
        sigma_to_eps=sigma_to_eps,
        target_epsilon=target_epsilon,
        threshold=threshold,
    )


def get_sigma_from_gdp(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: Union[float, int],
    threshold=1e-7,
    sigma_hi_init=1e1,
    sigma_lo_init=0.4,
) -> float:
    """Get noise multiplier σ for a given ε from f-DP accounting."""
    steps = epochs / sample_rate

    def sigma_to_eps(sigma):
        """Compute ε for a given σ based on f-DP."""
        eps, _ = _eps_from_gdp(
            sample_rate=sample_rate,
            sigma=sigma,
            steps=steps,
            delta=target_delta,
        )
        return eps

    return _get_sigma_with_target_epsilon(
        sigma_hi_init=sigma_hi_init,
        sigma_lo_init=sigma_lo_init,
        sigma_to_eps=sigma_to_eps,
        target_epsilon=target_epsilon,
        threshold=threshold,
    )


def _get_sigma_with_target_epsilon(
    sigma_hi_init: float,
    sigma_lo_init: float,
    sigma_to_eps: Callable,
    target_epsilon: float,
    threshold,
) -> float:
    """Core logic for binary searching σ given ε.."""
    sigma_hi = sigma_hi_init
    sigma_lo = sigma_lo_init

    # Find an appropriate region for binary search.
    while True:
        eps = sigma_to_eps(sigma_hi)
        if eps < target_epsilon:
            break
        sigma_hi *= 2

    while True:
        eps = sigma_to_eps(sigma_lo)
        if eps > target_epsilon:
            break
        sigma_lo /= 2

    # Binary search.
    while sigma_hi - sigma_lo > threshold:
        sigma = (sigma_hi + sigma_lo) / 2
        eps = sigma_to_eps(sigma)
        if eps < target_epsilon:
            sigma_hi = sigma
        else:
            sigma_lo = sigma

    # Conservative estimate.
    return sigma_hi


def _eps_from_rdp(
    sample_rate,
    sigma,
    steps,
    delta,
    alphas=DEFAULT_ALPHAS,
    **_,
):
    """Get the ε in (ε, δ)-DP from Renyi-DP accounting."""
    # This is based on Poisson sampling in https://arxiv.org/pdf/1908.10530.pdf
    rdp = rdp_accounting.compute_rdp(
        q=sample_rate, noise_multiplier=sigma, steps=steps, orders=alphas
    )
    # (ε, α)
    eps, alpha = rdp_accounting.get_privacy_spent(
        orders=alphas, rdp=rdp, delta=delta
    )
    return eps, alpha


def _compute_eps_cks(orders, rdp, delta):
    """Compute epsilon given a list of RDP values and target delta.
    Args:
      orders: An array (or a scalar) of orders.
      rdp: A list (or a scalar) of RDP guarantees.
      delta: The target delta.
    Returns:
      Pair of (eps, optimal_order).
    Raises:
      ValueError: If input is malformed.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if delta <= 0:
        raise ValueError("Privacy failure probability bound delta must be >0.")
    if len(orders_vec) != len(rdp_vec):
        raise ValueError("Input lists must have the same length.")

    # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
    #   eps = min( rdp_vec - math.log(delta) / (orders_vec - 1) )

    # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4).
    # Also appears in https://arxiv.org/abs/2001.05990 Equation 20 (in v1).
    eps_vec = []
    for (a, r) in zip(orders_vec, rdp_vec):
        if a < 1: raise ValueError("Renyi divergence order must be >=1.")
        if r < 0: raise ValueError("Renyi divergence must be >=0.")

        if delta ** 2 + math.expm1(-r) >= 0:
            # In this case, we can simply bound via KL divergence:
            # delta <= sqrt(1-exp(-KL)).
            eps = 0  # No need to try further computation if we have eps = 0.
        elif a > 1.01:
            # This bound is not numerically stable as alpha->1.
            # Thus we have a min value of alpha.
            # The bound is also not useful for small alpha, so doesn't matter.
            eps = r + math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
        else:
            # In this case we can't do anything. E.g., asking for delta = 0.
            eps = np.inf
        eps_vec.append(eps)

    idx_opt = np.argmin(eps_vec)
    return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]


# @formatter:off
def _eps_from_rdp_cks(
    sample_rate,
    sigma,
    steps,
    delta,
    alphas=DEFAULT_ALPHAS,
    **_,
):
    """Compute RDP as usual, but the conversion to (ε, δ)-DP is based on result by Canonne, Kamath, Steinke.
    Code from https://github.com/tensorflow/privacy/blob/5f07198b66b3617b22609db983926e3ba97cd905/tensorflow_privacy/privacy/analysis/rdp_accountant.py#L237
    """
    rdp = rdp_accounting.compute_rdp(
        q=sample_rate, noise_multiplier=sigma, steps=steps, orders=alphas
    )
    # (ε, α)
    eps, alpha = _compute_eps_cks(orders=alphas, rdp=rdp, delta=delta)
    return eps, alpha
# @formatter:on


def _eps_from_gdp(
    sample_rate,
    sigma,
    steps,
    delta,
    mode="poisson",
    **_,
):
    """Get the ε in (ε, δ)-DP from f-DP accounting."""
    epochs = steps * sample_rate
    if mode == "poisson":
        eps_fn = gdp_accounting.compute_eps_poisson
        mu_fn = gdp_accounting.compute_mu_poisson
    else:
        eps_fn = gdp_accounting.compute_eps_uniform
        mu_fn = gdp_accounting.compute_mu_uniform

    eps = eps_fn(
        epochs=epochs,
        noise_multi=sigma,
        delta=delta,
        sample_rate=sample_rate,
    )
    mu = mu_fn(
        epochs=epochs,
        noise_multi=sigma,
        sample_rate=sample_rate,
    )
    return eps, mu
