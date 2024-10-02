from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import scipy.linalg
import torch
from moving_targets.masters.backends import Backend
from scipy.optimize import NonlinearConstraint, minimize
from scipy.stats import pearsonr

from items.indicators.indicator import CopulaIndicator, DeclarativeIndicator, RegularizerIndicator

DEGREE: int = 5
"""Default degree for kernel-based indicators."""

TOL: float = 1e-2
"""The tolerance used when checking for linear dependencies and when solving the optimization problem."""

MAX_SIZE: int = 25000
"""The maximal size after which using least squares regression becomes inconvenient."""


@dataclass(frozen=True, eq=False)
class KernelBasedIndicator(RegularizerIndicator, CopulaIndicator, ABC):
    """Interface for an indicator that uses polynomial kernel expansions to compute the correlation value."""

    @staticmethod
    def kernel(v, degree: int, use_torch: bool):
        """Computes the kernel of the given vector with the given degree and using either numpy or torch as backend."""
        if use_torch:
            return torch.stack([v ** d - torch.mean(v ** d) for d in np.arange(degree) + 1], dim=1)
        else:
            return np.stack([v ** d - np.mean(v ** d) for d in np.arange(degree) + 1], axis=1)


@dataclass(frozen=True, eq=False)
class KernelBasedHGR(KernelBasedIndicator):
    """Kernel-based HGR interface."""

    @property
    @abstractmethod
    def degree_a(self) -> int:
        """The kernel degree for the first variable."""
        pass

    @property
    @abstractmethod
    def degree_b(self) -> int:
        """The kernel degree for the first variable."""
        pass

    def copulas(self, a: np.ndarray, b: np.ndarray, experiment: Any) -> Tuple[np.ndarray, np.ndarray]:
        # center the kernels with respect to the training data
        f = self.kernel(v=a, degree=self.degree_a, use_torch=False)
        g = self.kernel(v=b, degree=self.degree_b, use_torch=False)
        return f @ np.array(experiment['alpha']), g @ np.array(experiment['beta'])

    @staticmethod
    def _get_linearly_independent(f: np.ndarray, g: np.ndarray) -> Tuple[List[int], List[int]]:
        """Returns the list of indices of those columns that are linearly independent to other ones."""
        n, dx = f.shape
        _, dy = g.shape
        d = dx + dy
        # build a new matrix [ 1 | F_1 | G_1 | F_2 | G_2 | ... ]
        #   - this order is chosen so that lower grades are preferred in case of linear dependencies
        #   - the F and G indices are built depending on which kernel has the higher degree
        if dx < dy:
            f_indices = [2 * i + 1 for i in range(dx)]
            g_indices = [2 * i + 2 for i in range(dx)] + [i + 1 for i in range(2 * dx, d)]
        else:
            f_indices = [2 * i + 1 for i in range(dy)] + [i + 1 for i in range(2 * dy, d)]
            g_indices = [2 * i + 2 for i in range(dy)]
        fg_bias = np.ones((len(f), d + 1))
        fg_bias[:, f_indices] = f
        fg_bias[:, g_indices] = g
        # compute the QR factorization and retrieve the R matrix
        #   - get the diagonal of R
        #   - if a null value is found, it means that the respective column is linearly dependent to other columns
        # noinspection PyUnresolvedReferences
        r = scipy.linalg.qr(fg_bias, mode='r')[0]
        r = np.abs(np.diag(r))
        # eventually, retrieve the indices to be set to zero:
        #   - create a range going from 0 to degree - 1
        #   - mask it by selecting all those value in the diagonal that are smaller than the tolerance
        #   - finally exclude the first value in both cases since their linear dependence might be caused by a
        #      deterministic dependency in the data which we don't want to exclude
        f_indices = np.arange(dx)[r[f_indices] <= TOL][1:]
        g_indices = np.arange(dy)[r[g_indices] <= TOL][1:]
        return ([idx for idx in range(dx) if idx not in f_indices],
                [idx for idx in range(dy) if idx not in g_indices])

    @staticmethod
    def _higher_order_coefficients(f: np.ndarray,
                                   g: np.ndarray,
                                   a0: Optional[np.ndarray],
                                   b0: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the kernel-based hgr for higher order degrees."""
        degree_x, degree_y = f.shape[1], g.shape[1]
        # retrieve the indices of the linearly dependent columns and impose a linear constraint so that the respective
        # weight is null for all but the first one (this processing step allow to avoid degenerate cases when the
        # matrix is not full rank)
        f_indices, g_indices = KernelBasedHGR._get_linearly_independent(f=f, g=g)
        f_slim = f[:, f_indices]
        g_slim = g[:, g_indices]
        n, dx = f_slim.shape
        _, dy = g_slim.shape
        d = dx + dy
        fg = np.concatenate((f_slim, -g_slim), axis=1)

        # define the function to optimize as the least square problem:
        #   - func:   || F @ alpha - G @ beta ||_2^2 =
        #           =   (F @ alpha - G @ beta) @ (F @ alpha - G @ beta)
        #   - grad:   [ 2 * F.T @ (F @ alpha - G @ beta) | -2 * G.T @ (F @ alpha - G @ beta) ] =
        #           =   2 * [F | -G].T @ (F @ alpha - G @ beta)
        #   - hess:   [  2 * F.T @ F | -2 * F.T @ G ]
        #             [ -2 * G.T @ F |  2 * G.T @ G ] =
        #           =    2 * [F  -G].T @ [F  -G]
        #
        # plus, add the lasso regularization
        #   - func:     norm_1([alpha, beta])
        #   - grad:   [ sign(alpha) | sign(beta) ]
        #   - hess:   [      0      |      0     ]
        #             [      0      |      0     ]
        def _fun(inp):
            alp, bet = inp[:dx], inp[dx:]
            diff = f_slim @ alp - g_slim @ bet
            obj_func = diff @ diff
            obj_grad = 2 * fg.T @ diff
            # pen_func = np.abs(inp).sum()
            # pen_grad = np.sign(inp)
            return obj_func, obj_grad

        fun_hess = 2 * fg.T @ fg

        # define the constraint
        #   - func:   var(G @ beta) --> = 1
        #   - grad: [ 0 | 2 * G.T @ G @ beta / n ]
        #   - hess: [ 0 |         0       ]
        #           [ 0 | 2 * G.T @ G / n ]
        cst_hess = np.zeros(shape=(d, d), dtype=float)
        cst_hess[dx:, dx:] = 2 * g_slim.T @ g_slim / n
        constraint = NonlinearConstraint(
            fun=lambda inp: np.var(g_slim @ inp[dx:], ddof=0),
            jac=lambda inp: np.concatenate(([0] * dx, 2 * g_slim.T @ g_slim @ inp[dx:] / n)),
            hess=lambda *_: cst_hess,
            lb=1,
            ub=1
        )
        # if no guess is provided, set the initial point as [ 1 / std(F @ 1) | 1 / std(G @ 1) ] then solve the problem
        a0 = np.ones(dx) / f_slim.sum(axis=1).std(ddof=0) if a0 is None else a0[f_indices]
        b0 = np.ones(dy) / g_slim.sum(axis=1).std(ddof=0) if b0 is None else b0[g_indices]
        x0 = np.concatenate((a0, b0))
        # noinspection PyTypeChecker
        s = minimize(
            _fun,
            jac=True,
            hess=lambda *_: fun_hess,
            x0=x0,
            constraints=[constraint],
            method='trust-constr',
            tol=TOL,
        )
        # reconstruct alpha and beta by adding zeros wherever the indices were not considered
        alpha = np.zeros(degree_x)
        alpha[f_indices] = s.x[:dx]
        beta = np.zeros(degree_y)
        beta[g_indices] = s.x[dx:]
        return alpha, beta

    @staticmethod
    def _compute_numpy(a: np.ndarray,
                       b: np.ndarray,
                       degree_a: int,
                       degree_b: int,
                       a0: Optional[np.ndarray],
                       b0: Optional[np.ndarray],
                       lstsq: bool) -> Tuple[float, List[float], List[float]]:
        """Computes HGR using numpy as backend and returns the correlation along with alpha and beta."""
        # handle degenerate case of constant vectors
        std_a = a.std(ddof=0)
        std_b = b.std(ddof=0)
        if std_a == 0.0 or std_b == 0.0:
            return 0.0, [np.nan] * degree_a, [np.nan] * degree_b
        # build the kernel matrices
        f = KernelBasedHGR.kernel(a, degree=degree_a, use_torch=False)
        g = KernelBasedHGR.kernel(b, degree=degree_b, use_torch=False)
        # handle trivial or simpler cases:
        #  - if both degrees are 1, simply compute the projected vectors as standardized original vectors
        #  - if one degree is 1, standardize that vector and compute the other's coefficients using lstsq
        #  - if no degree is 1, use the optimization routine and compute the projected vectors from the coefficients
        if degree_a == 1 and degree_b == 1:
            alpha, beta = np.ones(1), np.ones(1)
        elif degree_a == 1 and lstsq:
            alpha = np.ones(1) / std_a
            beta, _, _, _ = np.linalg.lstsq(g, f[:, 0] / std_a, rcond=None)
        elif degree_b == 1 and lstsq:
            beta = np.ones(1) / std_b
            alpha, _, _, _ = np.linalg.lstsq(f, g[:, 0] / std_b, rcond=None)
        else:
            alpha, beta = KernelBasedHGR._higher_order_coefficients(f=f, g=g, a0=a0, b0=b0)
        fa = f @ alpha
        gb = g @ beta
        correlation, _ = pearsonr(fa, gb)
        alpha = alpha / fa.std(ddof=0)
        beta = beta / gb.std(ddof=0)
        return abs(float(correlation)), [float(a) for a in alpha], [float(b) for b in beta]

    @staticmethod
    def _compute_torch(a: torch.Tensor,
                       b: torch.Tensor,
                       degree_a: int,
                       degree_b: int,
                       a0: Optional[np.ndarray],
                       b0: Optional[np.ndarray],
                       lstsq: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes HGR using numpy as backend and returns the correlation (without alpha and beta)."""

        def standardize(t: torch.Tensor) -> torch.Tensor:
            t_std, t_mean = torch.std_mean(t, correction=0)
            return (t - t_mean) / t_std

        # handle degenerate case of constant vectors
        std_a = a.std(correction=0)
        std_b = b.std(correction=0)
        if std_a == 0.0 or std_b == 0.0:
            return torch.tensor(0.0), torch.ones(degree_a) * np.nan, torch.ones(degree_b) * np.nan
        # build the kernel matrices
        f = KernelBasedHGR.kernel(a, degree=degree_a, use_torch=True)
        g = KernelBasedHGR.kernel(b, degree=degree_b, use_torch=True)
        # handle trivial or simpler cases:
        #  - if both degrees are 1, simply compute the projected vectors as standardized original vectors
        #  - if one degree is 1, standardize that vector and compute the other's coefficients using lstsq
        #  - if no degree is 1, use the optimization routine and compute the projected vectors from the coefficients
        alpha, beta = torch.ones(1, dtype=f.dtype), torch.ones(1, dtype=g.dtype)
        if degree_a == 1 and degree_b == 1:
            fa = standardize(a)
            gb = standardize(b)
        elif degree_a == 1 and lstsq:
            # the 'gelsd' driver allows to have both more precise and more reproducible results
            fa = standardize(a)
            beta, _, _, _ = torch.linalg.lstsq(g, fa, driver='gelsd')
            gb = standardize(g @ beta)
        elif degree_b == 1 and lstsq:
            # the 'gelsd' driver allows to have both more precise and more reproducible results
            gb = standardize(b)
            alpha, _, _, _ = torch.linalg.lstsq(f, gb, driver='gelsd')
            fa = standardize(f @ alpha)
        else:
            alpha, beta = KernelBasedHGR._higher_order_coefficients(
                f=f.numpy(force=True),
                g=g.numpy(force=True),
                a0=a0,
                b0=b0
            )
            alpha = torch.tensor(alpha, dtype=f.dtype)
            beta = torch.tensor(beta, dtype=g.dtype)
            fa = standardize(f @ alpha)
            gb = standardize(g @ beta)
        # return the correlation as the absolute value of the vector product (since the vectors are standardized)
        return torch.abs(torch.mean(fa * gb)), alpha, beta


@dataclass(frozen=True, eq=False)
class DoubleKernelHGR(KernelBasedHGR):
    """Kernel-based HGR computed by solving a constrained least square problem using a minimization solver."""

    degree_a: int = field(init=True, default=DEGREE)
    """The kernel degree for the first variable."""

    degree_b: int = field(init=True, default=DEGREE)
    """The kernel degree for the second variable."""

    @property
    def name(self) -> str:
        return 'kb'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name, degree_a=self.degree_a, degree_b=self.degree_b)

    def correlation(self, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        # do not use x0 when computing correlations for correlation experiments
        correlation, alpha, beta = KernelBasedHGR._compute_numpy(
            a=a,
            b=b,
            degree_a=self.degree_a,
            degree_b=self.degree_b,
            a0=None,
            b0=None,
            lstsq=False
        )
        return dict(correlation=float(correlation), alpha=alpha, beta=beta)

    def regularizer(self, a: torch.Tensor, b: torch.Tensor, threshold: float, kwargs: Dict[str, Any]) -> torch.Tensor:
        # set default kwargs in case they are not set (and overwrite them for next steps)
        kwargs['a0'] = kwargs.get('a0', None)
        kwargs['b0'] = kwargs.get('b0', None)
        correlation, alpha, beta = KernelBasedHGR._compute_torch(
            a=a,
            b=b,
            degree_a=self.degree_a,
            degree_b=self.degree_b,
            lstsq=False,
            **kwargs
        )
        # eventually, replace a0/b0 in the arguments with the new value for the next training step
        kwargs['a0'] = alpha.numpy(force=True)
        kwargs['b0'] = beta.numpy(force=True)
        # return the regularizer
        return torch.maximum(torch.zeros(1), correlation - threshold)


@dataclass(frozen=True, eq=False)
class SingleKernelHGR(KernelBasedHGR):
    """Kernel-based HGR computed using one kernel only for both variables and then taking the maximal correlation."""

    degree: int = field(init=True, default=DEGREE)
    """The kernel degree for the variables."""

    lstsq: Optional[bool] = field(init=True, default=None)
    """Whether to use lstsq or global optimization to compute the gradients (if None, sets the value automatically)."""

    @property
    def name(self) -> str:
        return 'sk'

    @property
    def degree_a(self) -> int:
        return self.degree

    @property
    def degree_b(self) -> int:
        return self.degree

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name, degree=self.degree, lstsq=self.lstsq)

    def correlation(self, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        # compute single-kernel correlations along with kernels
        # (x0 is not used when either one degree is 1, so None is always passed)
        lstsq = (len(a) < MAX_SIZE) if self.lstsq is None else self.lstsq
        correlation_a, alpha_a, beta_a = KernelBasedHGR._compute_numpy(
            a=a,
            b=b,
            degree_a=self.degree,
            degree_b=1,
            a0=None,
            b0=None,
            lstsq=lstsq
        )
        correlation_b, alpha_b, beta_b = KernelBasedHGR._compute_numpy(
            a=a,
            b=b,
            degree_a=1,
            degree_b=self.degree,
            a0=None,
            b0=None,
            lstsq=lstsq
        )
        # choose the best correlation and return
        if correlation_a > correlation_b:
            correlation = correlation_a
            alpha = alpha_a
            beta = beta_a + [0] * (self.degree - 1)
        else:
            correlation = correlation_b
            alpha = alpha_b + [0] * (self.degree - 1)
            beta = beta_b
        return dict(correlation=float(correlation), alpha=alpha, beta=beta)

    def regularizer(self, a: torch.Tensor, b: torch.Tensor, threshold: float, kwargs: Dict[str, Any]) -> torch.Tensor:
        # compute single-kernel correlations
        # (x0 is not used when either one degree is 1, so None is always passed)
        lstsq = (len(a) < MAX_SIZE) if self.lstsq is None else self.lstsq
        hgr_a, _, _ = KernelBasedHGR._compute_torch(
            a=a,
            b=b,
            degree_a=self.degree,
            degree_b=1,
            a0=None,
            b0=None,
            lstsq=lstsq
        )
        hgr_b, _, _ = KernelBasedHGR._compute_torch(
            a=a,
            b=b,
            degree_a=1,
            degree_b=self.degree,
            a0=None,
            b0=None,
            lstsq=lstsq
        )
        # return the regularizer on the maximal correlation
        correlation = torch.maximum(hgr_a, hgr_b)
        return torch.maximum(torch.zeros(1), correlation - threshold)


@dataclass(frozen=True, eq=False)
class KernelBasedGeDI(KernelBasedIndicator, DeclarativeIndicator):
    degree: int = field(init=True, default=DEGREE)
    """The kernel degree for the variable."""

    fine_grained: bool = field(init=True, default=False)
    """Whether to use the fine-grained or coarse-grained formulation of the constraint."""

    @property
    def name(self) -> str:
        return 'gedi'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name, degree=self.degree, fine_grained=self.fine_grained)

    def correlation(self, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        f = self.kernel(v=a, degree=self.degree, use_torch=False)
        alpha, _, _, _ = np.linalg.lstsq(f, b - b.mean(), rcond=None)
        gedi = np.std(f @ alpha, ddof=0) / a.std(ddof=0)
        return dict(correlation=float(gedi), alpha=[float(a) for a in alpha])

    def regularizer(self, a: torch.Tensor, b: torch.Tensor, threshold: float, kwargs: Dict[str, Any]) -> torch.Tensor:
        # use the least-square operator to compute the \tilde{alpha} vector
        f = self.kernel(v=a, degree=self.degree, use_torch=True)
        alpha, _, _, _ = torch.linalg.lstsq(f, b - b.mean(), driver='gelsd')
        if self.fine_grained:
            # in the fine-grained constraint, we want a regularizer:
            #    [ max{ 0, | \tilde{alpha}_1 | - threshold }, | \tilde{alpha}_2 |, ..., | \tilde{alpha}_h | ]
            alpha = torch.abs(alpha)
            thresholds = torch.tensor([threshold] + [0] * (len(alpha) - 1))
            return torch.maximum(torch.zeros_like(alpha), alpha - thresholds)
        else:
            # in the coarse-grained constraints, we simply impose a standard regularizer on the GeDI value, which is
            # computed as std(P_a^h @ \tilde{alpha}) / std(a)
            correlation = torch.std(f @ alpha, correction=0) / a.std(correction=0)
            return torch.maximum(torch.zeros(1), correlation - threshold)

    def formulation(self, a: np.ndarray, b: np.ndarray, threshold: float, backend: Backend) -> Any:
        var_a = a.var(ddof=0)
        if self.fine_grained:
            cov_ab = backend.mean(a * b) - a.mean() * backend.mean(b)
            # impose the fine-grained zero constraints on \alpha_i, for each i in {2, ..., h}, i.e.:
            #    cov(a^d, a) * cov(a, b) == var(a) * cov(a^d, v)
            for d in np.arange(1, self.degree) + 1:
                ad = a ** d
                cov_ada = np.cov(ad, a, ddof=0)[0, 1]
                cov_adb = backend.mean(ad * b) - ad.mean() * backend.mean(b)
                backend.add_constraint(cov_ab * cov_ada == cov_adb * var_a)
            # the value of GeDI(a, b) is computed as | cov(a, b) | / var(a)
            # since we want GeDI(a, b) <= threshold, we can avoid a division by rewriting it as:
            #    | cov(a, b) | <= threshold * var(a)
            # and, to avoid computing the absolute value, we can impose:
            #    -threshold * var(a) <= cov(a, b) <= threshold * var(a)
            backend.add_constraints([cov_ab >= -threshold * var_a, cov_ab <= threshold * var_a])
        else:
            # we build a vector of continuous variables for <alpha> and center the target vector <b>
            alpha = backend.add_continuous_variables(self.degree, name='alpha')
            b = b - backend.mean(b, aux=True)
            # we build the polynomial kernel and impose the equations of the linear system
            f = self.kernel(v=a, degree=self.degree, use_torch=False)
            ftf_alpha = backend.dot(f.T @ f, alpha)
            left_hand_sides = np.atleast_1d(ftf_alpha)
            right_hand_sides = np.atleast_1d(backend.dot(f.T, b))
            backend.add_constraints([lhs == rhs for lhs, rhs in zip(left_hand_sides, right_hand_sides)])
            # eventually, we enforce our constraint on the squared term:
            #     var(f @ alpha) <= threshold ** 2 * var(a)
            # where var(f @ alpha) = alpha^T @ (f^T @ f @ alpha) / n, with the (f^T @ f @ alpha) term already computed
            var_fa = backend.dot(alpha.T, ftf_alpha) / len(a)
            backend.add_constraint(var_fa <= threshold ** 2 * var_a)

    def copulas(self, a: np.ndarray, b: np.ndarray, experiment: Any) -> Tuple[np.ndarray, np.ndarray]:
        f = self.kernel(v=a, degree=self.degree, use_torch=False)
        return f @ np.array(experiment['alpha']), b - b.mean()
