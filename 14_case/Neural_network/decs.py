"""Differentiable fixed-PV equality completion for the IEEE 14-bus case.

The surrogate follows the paper's fixed-PV state construction. Its input is
the power-flow specification

    rho = [p_PV, V_PV_target, V_ref, p_PQ, q_PQ],

and its output is

    chi = [theta_PV, theta_PQ, V_PQ].

PV and reference voltage magnitudes are fixed by the projected neural controls.
Only ``chi`` is learned. Generator reactive powers, reference-bus powers, and
branch flows are reconstructed with differentiable AC-network equations.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Data_generation.case14_pglib import load_case14  # noqa: E402


class EqualityCompletionSurrogate(nn.Module):
    """Map normalized PF specifications to normalized voltage states.

    Parameters
    ----------
    input_dim : int, default=27
        Number of PF-specification features.
    hidden_dims : tuple[int, int], default=(128, 128)
        Widths of the two ReLU hidden layers.
    output_dim : int, default=22
        Number of dependent fixed-PV voltage-state variables.
    """

    def __init__(
        self,
        input_dim: int = 27,
        hidden_dims: tuple[int, int] = (128, 128),
        output_dim: int = 22,
    ):
        """Build the ReLU MLP with the dimensions documented by the class."""
        super().__init__()
        if input_dim < 1 or output_dim < 1 or any(width < 1 for width in hidden_dims):
            raise ValueError("all DECS layer dimensions must be positive")
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, rho_normalized: torch.Tensor) -> torch.Tensor:
        """Predict normalized voltage states from normalized specifications.

        Parameters
        ----------
        rho_normalized : torch.Tensor, shape (..., 27)
            Standardized PF specifications.

        Returns
        -------
        torch.Tensor, shape (..., 22)
            Standardized nonreference angles and voltage magnitudes.
        """
        return self.network(rho_normalized)


class ACReconstruction(nn.Module):
    """Assemble PF specifications and reconstruct dependent AC quantities."""

    def __init__(self):
        """Load the local PGLib IEEE-14 topology as fixed tensor buffers."""
        super().__init__()
        case = load_case14()
        real = torch.float32
        self.n_bus = case.n_bus
        self.n_gen = case.n_gen
        self.base_mva = float(case.base_mva)
        self.reference_generator = case.reference_generator
        self.n_free_pg = len(case.nonreference_active_generators)
        self.free_dim = self.n_free_pg + len(case.voltage_control_buses)
        self.rho_dim = 2 * len(case.pv_buses) + 1 + 2 * len(case.pq_buses)
        self.chi_dim = len(case.pv_buses) + 2 * len(case.pq_buses)

        self.register_buffer("g", torch.tensor(case.ybus.real, dtype=real))
        self.register_buffer("b", torch.tensor(case.ybus.imag, dtype=real))
        self.register_buffer("gs", torch.tensor(case.bus[:, 4], dtype=real))
        self.register_buffer("bs", torch.tensor(case.bus[:, 5], dtype=real))
        self.register_buffer("load_buses", torch.tensor(case.load_buses, dtype=torch.long))
        self.register_buffer("active", torch.tensor(case.active_generators, dtype=torch.long))
        self.register_buffer(
            "free_pg", torch.tensor(case.nonreference_active_generators, dtype=torch.long),
        )
        self.register_buffer("gen_buses", torch.tensor(case.generator_buses, dtype=torch.long))
        self.register_buffer("pv", torch.tensor(case.pv_buses, dtype=torch.long))
        self.register_buffer("pq", torch.tensor(case.pq_buses, dtype=torch.long))
        self.register_buffer(
            "voltage_buses", torch.tensor(case.voltage_control_buses, dtype=torch.long),
        )
        self.reference_bus = case.reference_bus

        branch = case.branch
        self.register_buffer("fbus", torch.tensor(branch[:, 0].astype(int) - 1, dtype=torch.long))
        self.register_buffer("tbus", torch.tensor(branch[:, 1].astype(int) - 1, dtype=torch.long))
        self.register_buffer("rate", torch.tensor(branch[:, 5], dtype=real))
        self.register_buffer("angle_min", torch.tensor(np.deg2rad(branch[:, 11]), dtype=real))
        self.register_buffer("angle_max", torch.tensor(np.deg2rad(branch[:, 12]), dtype=real))
        self.register_buffer("yff_real", torch.tensor(case.yff.real, dtype=real))
        self.register_buffer("yff_imag", torch.tensor(case.yff.imag, dtype=real))
        self.register_buffer("yft_real", torch.tensor(case.yft.real, dtype=real))
        self.register_buffer("yft_imag", torch.tensor(case.yft.imag, dtype=real))
        self.register_buffer("ytf_real", torch.tensor(case.ytf.real, dtype=real))
        self.register_buffer("ytf_imag", torch.tensor(case.ytf.imag, dtype=real))
        self.register_buffer("ytt_real", torch.tensor(case.ytt.real, dtype=real))
        self.register_buffer("ytt_imag", torch.tensor(case.ytt.imag, dtype=real))

        self.register_buffer("pg_min", torch.tensor(case.gen[:, 9], dtype=real))
        self.register_buffer("pg_max", torch.tensor(case.gen[:, 8], dtype=real))
        self.register_buffer("qg_min", torch.tensor(case.gen[:, 4], dtype=real))
        self.register_buffer("qg_max", torch.tensor(case.gen[:, 3], dtype=real))
        self.register_buffer("vm_min", torch.tensor(case.bus[:, 12], dtype=real))
        self.register_buffer("vm_max", torch.tensor(case.bus[:, 11], dtype=real))
        self.register_buffer("cost_c2", torch.tensor(case.gencost[:, 4], dtype=real))
        self.register_buffer("cost_c1", torch.tensor(case.gencost[:, 5], dtype=real))
        self.register_buffer("cost_c0", torch.tensor(case.gencost[:, 6], dtype=real))

    def free_variable_bounds(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return physical bounds of ``[Pg_nonref, V_PV, V_ref]``.

        Parameters
        ----------
        batch_size : int
            Number of identical bound rows to return.

        Returns
        -------
        lower, upper : tuple[torch.Tensor, torch.Tensor], shape (batch_size, 6)
            Generator bounds in MW and voltage bounds in p.u.
        """
        lower = torch.cat([
            self.pg_min[self.free_pg].expand(batch_size, -1),
            self.vm_min[self.voltage_buses].expand(batch_size, -1),
        ], dim=1)
        upper = torch.cat([
            self.pg_max[self.free_pg].expand(batch_size, -1),
            self.vm_max[self.voltage_buses].expand(batch_size, -1),
        ], dim=1)
        return lower, upper

    def full_load(self, load: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand compressed load-bus values to all buses.

        Parameters
        ----------
        load : torch.Tensor, shape (batch, 11, 2)
            Active and reactive net loads ``[P,Q]`` in MW/Mvar.

        Returns
        -------
        pd, qd : tuple[torch.Tensor, torch.Tensor], shape (batch, 14)
            Full-bus active and reactive loads in MW/Mvar.
        """
        if load.ndim != 3 or load.shape[1:] != (len(self.load_buses), 2):
            raise ValueError(f"load must have shape (batch,{len(self.load_buses)},2)")
        pd = torch.zeros((len(load), self.n_bus), dtype=load.dtype, device=load.device)
        qd = torch.zeros_like(pd)
        pd[:, self.load_buses] = load[:, :, 0]
        qd[:, self.load_buses] = load[:, :, 1]
        return pd, qd

    def generator_active_power(self, u: torch.Tensor) -> torch.Tensor:
        """Assemble specified generator active powers before slack recovery.

        Parameters
        ----------
        u : torch.Tensor, shape (batch, 6)
            Free variables with active powers in MW and voltages in p.u.

        Returns
        -------
        torch.Tensor, shape (batch, 5)
            Generator active powers; the reference-generator entry is a
            placeholder and is replaced during AC reconstruction.
        """
        pg = self.pg_min.expand(len(u), -1).clone()
        pg[:, self.free_pg] = u[:, :self.n_free_pg]
        return pg

    def specification(self, u: torch.Tensor, load: torch.Tensor) -> torch.Tensor:
        """Construct the 27-dimensional PF specification ``rho``.

        Parameters
        ----------
        u : torch.Tensor, shape (batch, 6)
            ``[Pg_nonref, V_PV, V_ref]`` in MW and p.u.
        load : torch.Tensor, shape (batch, 11, 2)
            Net loads at nonzero-load buses in MW/Mvar.

        Returns
        -------
        torch.Tensor, shape (batch, 27)
            ``[p_PV, V_PV_target, V_ref, p_PQ, q_PQ]``. Power components are p.u.
            on the case base MVA and voltage components are p.u.
        """
        if u.ndim != 2 or u.shape[1] != self.free_dim:
            raise ValueError(f"u must have shape (batch,{self.free_dim})")
        pd, qd = self.full_load(load)
        return self._specification_from_full_load(u, pd, qd)

    def _specification_from_full_load(
        self,
        u: torch.Tensor,
        pd: torch.Tensor,
        qd: torch.Tensor,
    ) -> torch.Tensor:
        """Construct the PF specification from already expanded bus loads.

        Parameters
        ----------
        u : torch.Tensor, shape (batch, 6)
            Free variables in MW and p.u.
        pd, qd : torch.Tensor, shape (batch, 14)
            Full-bus active and reactive loads in MW/Mvar.

        Returns
        -------
        torch.Tensor, shape (batch, 27)
            Fixed-PV power-flow specification in the ordering used by DECS.
        """
        if u.ndim != 2 or u.shape[1] != self.free_dim:
            raise ValueError(f"u must have shape (batch,{self.free_dim})")
        expected = (len(u), self.n_bus)
        if tuple(pd.shape) != expected or tuple(qd.shape) != expected:
            raise ValueError(f"pd and qd must have shape {expected}")
        pg = self.generator_active_power(u)
        pgen_bus = torch.zeros_like(pd).scatter_add(
            1, self.gen_buses[None, :].expand(len(u), -1), pg,
        )
        p_pv = (pgen_bus[:, self.pv] - pd[:, self.pv]) / self.base_mva
        p_pq = -pd[:, self.pq] / self.base_mva
        q_pq = -qd[:, self.pq] / self.base_mva
        voltages = u[:, self.n_free_pg:]
        return torch.cat([
            p_pv,
            voltages[:, :len(self.pv)],
            voltages[:, len(self.pv):],
            p_pq,
            q_pq,
        ], dim=1)

    def voltage_state(self, u: torch.Tensor, chi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble full bus voltage magnitudes and phase angles.

        Parameters
        ----------
        u : torch.Tensor, shape (batch, 6)
            Free generator powers and controlled voltage magnitudes.
        chi : torch.Tensor, shape (batch, 22)
            ``[theta_PV, theta_PQ, V_PQ]`` in rad and p.u.

        Returns
        -------
        vm, va : tuple[torch.Tensor, torch.Tensor], shape (batch, 14)
            Full voltage magnitudes in p.u. and angles in rad.
        """
        if chi.ndim != 2 or chi.shape[1] != self.chi_dim:
            raise ValueError(f"chi must have shape (batch,{self.chi_dim})")
        n_pv, n_pq = len(self.pv), len(self.pq)
        vm = torch.ones((len(chi), self.n_bus), dtype=chi.dtype, device=chi.device)
        va = torch.zeros_like(vm)
        va[:, self.pv] = chi[:, :n_pv]
        va[:, self.pq] = chi[:, n_pv:n_pv + n_pq]
        vm[:, self.pq] = chi[:, n_pv + n_pq:]
        vm[:, self.voltage_buses] = u[:, self.n_free_pg:]
        return vm, va

    def _branch_power(
        self,
        vm: torch.Tensor,
        va: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate both-end branch powers for all states.

        Parameters
        ----------
        vm, va : torch.Tensor, shape (batch, 14)
            Voltage magnitudes in p.u. and phase angles in rad.

        Returns
        -------
        pf, qf, pt, qt : tuple[torch.Tensor, ...], shape (batch, n_branch)
            From- and to-end active/reactive powers in MW/Mvar.
        """
        vr, vi = vm * torch.cos(va), vm * torch.sin(va)
        vf_r, vf_i = vr[:, self.fbus], vi[:, self.fbus]
        vt_r, vt_i = vr[:, self.tbus], vi[:, self.tbus]
        iff_r, iff_i = self._complex_multiply(self.yff_real, self.yff_imag, vf_r, vf_i)
        ift_r, ift_i = self._complex_multiply(self.yft_real, self.yft_imag, vt_r, vt_i)
        itf_r, itf_i = self._complex_multiply(self.ytf_real, self.ytf_imag, vf_r, vf_i)
        itt_r, itt_i = self._complex_multiply(self.ytt_real, self.ytt_imag, vt_r, vt_i)
        if_r, if_i = iff_r + ift_r, iff_i + ift_i
        it_r, it_i = itf_r + itt_r, itf_i + itt_i
        pf = (vf_r * if_r + vf_i * if_i) * self.base_mva
        qf = (vf_i * if_r - vf_r * if_i) * self.base_mva
        pt = (vt_r * it_r + vt_i * it_i) * self.base_mva
        qt = (vt_i * it_r - vt_r * it_i) * self.base_mva
        return pf, qf, pt, qt

    def _nodal_injection_from_branch_power(
        self,
        vm: torch.Tensor,
        pf: torch.Tensor,
        qf: torch.Tensor,
        pt: torch.Tensor,
        qt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Accumulate branch and shunt powers into bus injections.

        Parameters
        ----------
        vm : torch.Tensor, shape (batch, 14)
            Bus voltage magnitudes in p.u.
        pf, qf, pt, qt : torch.Tensor, shape (batch, n_branch)
            Both-end branch powers in MW/Mvar.

        Returns
        -------
        p, q : tuple[torch.Tensor, torch.Tensor], shape (batch, 14)
            Bus-to-network active and reactive injections in MW/Mvar.
        """
        from_index = self.fbus[None, :].expand(len(vm), -1)
        to_index = self.tbus[None, :].expand(len(vm), -1)
        p = torch.zeros_like(vm).scatter_add(1, from_index, pf)
        p = p.scatter_add(1, to_index, pt)
        q = torch.zeros_like(vm).scatter_add(1, from_index, qf)
        q = q.scatter_add(1, to_index, qt)
        voltage_squared = vm.square()
        return p + voltage_squared * self.gs, q - voltage_squared * self.bs

    def nodal_injection(self, vm: torch.Tensor, va: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate sparse branch-accumulated AC nodal injections.

        Parameters
        ----------
        vm, va : torch.Tensor, shape (batch, 14)
            Voltage magnitudes in p.u. and phase angles in rad.

        Returns
        -------
        p, q : tuple[torch.Tensor, torch.Tensor], shape (batch, 14)
            Bus-to-network active and reactive injections in MW/Mvar.
        """
        return self._nodal_injection_from_branch_power(
            vm, *self._branch_power(vm, va),
        )

    @staticmethod
    def _complex_multiply(
        ar: torch.Tensor,
        ai: torch.Tensor,
        br: torch.Tensor,
        bi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multiply complex tensors represented by real and imaginary parts.

        Parameters
        ----------
        ar, ai : torch.Tensor
            Real and imaginary parts of the first factor.
        br, bi : torch.Tensor
            Real and imaginary parts of the second factor.

        Returns
        -------
        real, imag : tuple[torch.Tensor, torch.Tensor]
            Real and imaginary parts of the product.
        """
        return ar * br - ai * bi, ar * bi + ai * br

    def reconstruct(
        self,
        u: torch.Tensor,
        load: torch.Tensor,
        chi: torch.Tensor,
        rho: torch.Tensor | None = None,
        bus_load: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Recover all dependent generator and branch quantities.

        Parameters
        ----------
        u : torch.Tensor, shape (batch, 6)
            Free variables in MW and p.u.
        load : torch.Tensor, shape (batch, 11, 2)
            Net loads in MW/Mvar.
        chi : torch.Tensor, shape (batch, 22)
            Predicted voltage state in rad and p.u.
        rho : torch.Tensor or None, shape (batch, 27), default=None
            Optional specification already evaluated by DECS. ``None`` computes it.
        bus_load : tuple[torch.Tensor, torch.Tensor] or None, default=None
            Optional full-bus ``(pd, qd)`` tensors in MW/Mvar. ``None`` expands
            ``load`` inside this method.

        Returns
        -------
        dict[str, torch.Tensor]
            Full voltage state, generator powers, branch flows, and the 22
            power-balance residuals in p.u.
        """
        vm, va = self.voltage_state(u, chi)
        pd, qd = self.full_load(load) if bus_load is None else bus_load
        pf, qf, pt, qt = self._branch_power(vm, va)
        pnet, qnet = self._nodal_injection_from_branch_power(vm, pf, qf, pt, qt)
        pg = self.generator_active_power(u)
        ref_bus = int(self.gen_buses[self.reference_generator])
        pg[:, self.reference_generator] = pd[:, ref_bus] + pnet[:, ref_bus]
        qg = qd[:, self.gen_buses] + qnet[:, self.gen_buses]

        pgen_bus = torch.zeros_like(pd).scatter_add(
            1, self.gen_buses[None, :].expand(len(u), -1), pg,
        )
        p_spec = pgen_bus - pd
        q_spec = -qd
        residual = torch.cat([
            (pnet[:, torch.cat([self.pv, self.pq])] - p_spec[:, torch.cat([self.pv, self.pq])])
            / self.base_mva,
            (qnet[:, self.pq] - q_spec[:, self.pq]) / self.base_mva,
        ], dim=1)

        if rho is None:
            rho = self._specification_from_full_load(u, pd, qd)
        return {
            "rho": rho,
            "chi": chi,
            "pg": pg,
            "qg": qg,
            "vm": vm,
            "va": va,
            "pf": pf,
            "qf": qf,
            "pt": pt,
            "qt": qt,
            "balance_residual": residual,
            "pf_residual": residual.abs().amax(dim=1),
        }

    def remaining_constraint_sizes(self) -> dict[str, int]:
        """Return the per-scenario/time sizes of the reduced inequality blocks.

        Free nonreference-generator active powers and the reference voltage are
        omitted because the CSNG projection enforces their bounds. Fixed PV and
        reference voltages are omitted for the same reason. Thermal limits are
        checked at both physical ends of every branch.

        Returns
        -------
        dict[str, int]
            Widths of the reference-Pg, Qg, PQ-voltage, angle, and two-end
            thermal residual blocks after upper/lower expansion where needed.
        """
        n_branch = len(self.fbus)
        return {
            "pg": 2,
            "qg": 2 * self.n_gen,
            "voltage": 2 * len(self.pq),
            "angle": 2 * n_branch,
            "thermal": 2 * n_branch,
        }

    def remaining_constraint_scales(self) -> dict[str, torch.Tensor]:
        """Return the selected diagonal entries of ``D_g`` by constraint block.

        Returns
        -------
        dict[str, torch.Tensor]
            Dimensionful inverse-range scales in exactly the same block order
            as :meth:`constraint_violation`. Upper/lower or from/to entries use
            the same physical scale and are therefore repeated.
        """
        reference = self.reference_generator
        pg = (self.pg_max[reference] - self.pg_min[reference]).reciprocal().repeat(2)
        qg = (self.qg_max - self.qg_min).clamp_min(1e-12).reciprocal().repeat(2)
        voltage = (
            (self.vm_max[self.pq] - self.vm_min[self.pq])
            .clamp_min(1e-12).reciprocal().repeat(2)
        )
        angle = (
            (self.angle_max - self.angle_min)
            .clamp_min(1e-12).reciprocal().repeat(2)
        )
        thermal = self.rate.reciprocal().repeat(2)
        return {
            "pg": pg,
            "qg": qg,
            "voltage": voltage,
            "angle": angle,
            "thermal": thermal,
        }

    def constraint_violation(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return Eq. (22) residuals for the remaining inequalities.

        Parameters
        ----------
        state : dict[str, torch.Tensor]
            Output of :meth:`reconstruct`.

        Returns
        -------
        torch.Tensor, shape (batch, n_constraints)
            Positive dimensionless residuals for reference-generator active
            power, all generator reactive powers, all PQ-bus voltages, branch
            angles, and both-end thermal limits.  ``D_g`` uses the inverse
            physical operating range for two-sided limits and the inverse line
            rating for thermal limits, giving order-one violations when a
            quantity exceeds its admissible range by roughly one full range.
        """
        pg = state["pg"][:, self.reference_generator:self.reference_generator + 1]
        pg_min = self.pg_min[self.reference_generator:self.reference_generator + 1]
        pg_max = self.pg_max[self.reference_generator:self.reference_generator + 1]
        pg_scale = (pg_max - pg_min).clamp_min(1e-12)
        pg_residual = torch.cat([
            (pg - pg_max) / pg_scale,
            (pg_min - pg) / pg_scale,
        ], dim=1)
        q_scale = (self.qg_max - self.qg_min).clamp_min(1e-12)
        q_residual = torch.cat([
            (state["qg"] - self.qg_max) / q_scale,
            (self.qg_min - state["qg"]) / q_scale,
        ], dim=1)
        vm = state["vm"][:, self.pq]
        vm_min = self.vm_min[self.pq]
        vm_max = self.vm_max[self.pq]
        vm_scale = (vm_max - vm_min).clamp_min(1e-12)
        vm_residual = torch.cat([
            (vm - vm_max) / vm_scale,
            (vm_min - vm) / vm_scale,
        ], dim=1)
        angle = state["va"][:, self.fbus] - state["va"][:, self.tbus]
        angle_scale = (self.angle_max - self.angle_min).clamp_min(1e-12)
        angle_residual = torch.cat([
            (angle - self.angle_max) / angle_scale,
            (self.angle_min - angle) / angle_scale,
        ], dim=1)
        # A physical line can have different apparent power at its two ends
        # because of losses and charging, so both ends enforce the same rating.
        # torch.hypot has an undefined backward derivative at (P,Q)=(0,0),
        # which produces NaN gradients even when the inactive ReLU contributes
        # zero loss.  The 1e-12 MVA^2 regularizer changes apparent power by at
        # most 1e-6 MVA at zero while giving the origin a finite zero gradient.
        apparent_from = torch.sqrt(
            state["pf"].square() + state["qf"].square() + 1e-12
        )
        apparent_to = torch.sqrt(
            state["pt"].square() + state["qt"].square() + 1e-12
        )
        thermal_residual = torch.cat([
            apparent_from / self.rate - 1.0,
            apparent_to / self.rate - 1.0,
        ], dim=1)
        violation = torch.relu(torch.cat([
            pg_residual, q_residual, vm_residual, angle_residual, thermal_residual,
        ], dim=1))
        expected = sum(self.remaining_constraint_sizes().values())
        if violation.shape[1] != expected:
            raise RuntimeError(
                f"remaining inequality width {violation.shape[1]} does not match {expected}"
            )
        return violation

    def generation_cost(
        self,
        pg: torch.Tensor,
        generators: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate quadratic generation cost along the final Pg dimension.

        Parameters
        ----------
        pg : torch.Tensor, shape (..., n_selected_generators)
            Active-power decisions in MW.
        generators : torch.Tensor or None
            Generator-table indices associated with the final dimension.
            ``None`` means all generators in table order.

        Returns
        -------
        torch.Tensor, shape (...,)
            Sum of quadratic costs over the selected generators.
        """
        if generators is None:
            generators = torch.arange(self.n_gen, device=pg.device)
        generators = generators.to(device=pg.device)
        if pg.shape[-1] != len(generators):
            raise ValueError("Pg width must match the selected generator indices")
        c2 = self.cost_c2[generators]
        c1 = self.cost_c1[generators]
        c0 = self.cost_c0[generators]
        return (c2 * pg.square() + c1 * pg + c0).sum(dim=-1)


class DifferentiableEqualityCompletion(nn.Module):
    """Combine the frozen DECS network with exact AC reconstruction.

    Parameters
    ----------
    model : EqualityCompletionSurrogate
        Neural mapping in standardized coordinates.
    rho_mean, rho_std : torch.Tensor, shape (27,)
        Training-set input mean and standard deviation.
    chi_mean, chi_std : torch.Tensor, shape (22,)
        Training-set output mean and standard deviation.
    """

    def __init__(
        self,
        model: EqualityCompletionSurrogate,
        rho_mean: torch.Tensor,
        rho_std: torch.Tensor,
        chi_mean: torch.Tensor,
        chi_std: torch.Tensor,
    ):
        """Store the network, AC equations, and training normalization."""
        super().__init__()
        self.model = model
        self.physics = ACReconstruction()
        self.register_buffer("rho_mean", rho_mean.float())
        self.register_buffer("rho_std", rho_std.float().clamp_min(1e-8))
        self.register_buffer("chi_mean", chi_mean.float())
        self.register_buffer("chi_std", chi_std.float().clamp_min(1e-8))

    def predict_chi(self, rho: torch.Tensor) -> torch.Tensor:
        """Predict physical voltage states for unnormalized specifications."""
        normalized = (rho - self.rho_mean) / self.rho_std
        return self.model(normalized) * self.chi_std + self.chi_mean

    def forward(self, u: torch.Tensor, load: torch.Tensor) -> dict[str, torch.Tensor]:
        """Complete free variables and loads into differentiable AC states.

        Parameters
        ----------
        u : torch.Tensor, shape (batch, 6)
            Free variables in MW and p.u.
        load : torch.Tensor, shape (batch, 11, 2)
            Net loads in MW/Mvar.

        Returns
        -------
        dict[str, torch.Tensor]
            Reconstructed AC state and equality residual diagnostics.
        """
        bus_load = self.physics.full_load(load)
        rho = self.physics._specification_from_full_load(u, *bus_load)
        chi = self.predict_chi(rho)
        return self.physics.reconstruct(
            u, load, chi, rho=rho, bus_load=bus_load,
        )


def load_decs_checkpoint(path: Path, device: torch.device) -> DifferentiableEqualityCompletion:
    """Load a trained DECS checkpoint for differentiable inference.

    Parameters
    ----------
    path : pathlib.Path
        Checkpoint written by ``train_decs.py``.
    device : torch.device
        CPU or CUDA device used for inference.

    Returns
    -------
    DifferentiableEqualityCompletion
        Model and fixed reconstruction tensors on ``device``.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    expected_output_dim = ACReconstruction().chi_dim
    if int(checkpoint["output_dim"]) != expected_output_dim:
        raise ValueError(
            f"DECS checkpoint output_dim={checkpoint['output_dim']} is obsolete; "
            f"fixed-PV reconstruction requires {expected_output_dim} outputs. "
            "Regenerate the DECS dataset and retrain DECS."
        )
    model = EqualityCompletionSurrogate(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        output_dim=int(checkpoint["output_dim"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    normalization = checkpoint["normalization"]
    completion = DifferentiableEqualityCompletion(
        model,
        torch.as_tensor(normalization["rho_mean"]),
        torch.as_tensor(normalization["rho_std"]),
        torch.as_tensor(normalization["chi_mean"]),
        torch.as_tensor(normalization["chi_std"]),
    ).to(device)
    completion.eval()
    return completion
