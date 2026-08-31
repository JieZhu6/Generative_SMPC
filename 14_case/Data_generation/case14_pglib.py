"""Load and preprocess the PGLib IEEE 14-bus MATPOWER case.

The source of truth is ``pglib_opf_case14_ieee.m`` in this directory.  The
small parser below intentionally supports only the numeric MATPOWER fields
used by this research code (baseMVA, bus, gen, gencost, and branch), which
keeps the experiment independent of a MATLAB installation.
"""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


CASE_FILE = Path(__file__).with_name("pglib_opf_case14_ieee.m")


@dataclass(frozen=True)
class Case14:
    """Numerical data and admittances for the IEEE 14-bus system."""

    base_mva: float
    bus: np.ndarray
    gen: np.ndarray
    branch: np.ndarray
    gencost: np.ndarray
    ybus: np.ndarray
    yff: np.ndarray
    yft: np.ndarray
    ytf: np.ndarray
    ytt: np.ndarray

    @property
    def name(self) -> str:
        """Return the MATPOWER case name stored in dataset metadata."""
        return "pglib_opf_case14_ieee"

    @property
    def n_bus(self) -> int:
        """Return the number of online buses."""
        return len(self.bus)

    @property
    def n_gen(self) -> int:
        """Return the number of online generators and synchronous condensers."""
        return len(self.gen)

    @property
    def load_buses(self) -> np.ndarray:
        """Return zero-based indices of buses with nonzero P or Q demand."""
        return np.flatnonzero((self.bus[:, 2] != 0.0) | (self.bus[:, 3] != 0.0))

    @property
    def active_generators(self) -> np.ndarray:
        """Generators with a nonzero active-power operating range."""
        return np.flatnonzero(self.gen[:, 8] > self.gen[:, 9])

    @property
    def reactive_generators(self) -> np.ndarray:
        """Return generators with a nonzero reactive-power operating range."""
        return np.flatnonzero(self.gen[:, 3] > self.gen[:, 4])

    @property
    def reference_bus(self) -> int:
        """Return the unique zero-based reference-bus index."""
        refs = np.flatnonzero(self.bus[:, 1] == 3)
        if len(refs) != 1:
            raise ValueError(f"expected one reference bus, found {len(refs)}")
        return int(refs[0])

    @property
    def pv_buses(self) -> np.ndarray:
        """Return zero-based PV-bus indices."""
        return np.flatnonzero(self.bus[:, 1] == 2)

    @property
    def pq_buses(self) -> np.ndarray:
        """Return zero-based PQ-bus indices."""
        return np.flatnonzero(self.bus[:, 1] == 1)

    @property
    def generator_buses(self) -> np.ndarray:
        """Return each generator's zero-based bus index in generator-table order."""
        return self.gen[:, 0].astype(int) - 1

    @property
    def reference_generator(self) -> int:
        """Return the generator-table index of the reference-bus generator."""
        generators = np.flatnonzero(self.generator_buses == self.reference_bus)
        if len(generators) != 1:
            raise ValueError(
                f"expected one generator at the reference bus, found {len(generators)}"
            )
        return int(generators[0])

    @property
    def nonreference_active_generators(self) -> np.ndarray:
        """Return adjustable active generators excluding the reference generator."""
        return self.active_generators[self.active_generators != self.reference_generator]

    @property
    def voltage_control_buses(self) -> np.ndarray:
        """PV buses followed by the reference bus, matching the paper's u."""
        return np.r_[self.pv_buses, self.reference_bus]


def _numeric_matrix(source: str, field: str) -> np.ndarray:
    """Parse one numeric MATPOWER matrix ``field`` from MATLAB source text."""
    match = re.search(rf"mpc\.{field}\s*=\s*\[(.*?)\];", source, flags=re.S)
    if match is None:
        raise ValueError(f"missing mpc.{field} in {CASE_FILE.name}")
    rows = []
    matrix_text = re.sub(r"%[^\r\n]*", "", match.group(1))
    for raw_row in matrix_text.split(";"):
        row = raw_row.strip()
        if row:
            rows.append(np.fromstring(row, sep=" ", dtype=float))
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError(f"invalid rectangular matrix mpc.{field}")
    return np.vstack(rows)


def _network_admittance(base_mva: float, bus: np.ndarray, branch: np.ndarray):
    """Build Ybus and both-end branch admittances from MATPOWER arrays."""
    n_bus, n_branch = len(bus), len(branch)
    ybus = np.zeros((n_bus, n_bus), dtype=complex)
    yff = np.empty(n_branch, dtype=complex)
    yft = np.empty(n_branch, dtype=complex)
    ytf = np.empty(n_branch, dtype=complex)
    ytt = np.empty(n_branch, dtype=complex)

    for ell, row in enumerate(branch):
        i, j = int(row[0]) - 1, int(row[1]) - 1
        y = 1.0 / complex(row[2], row[3])
        charging = 1j * row[4] / 2.0
        tap_magnitude = row[8] if row[8] != 0.0 else 1.0
        tap = tap_magnitude * np.exp(1j * np.deg2rad(row[9]))
        yff[ell] = (y + charging) / abs(tap) ** 2
        yft[ell] = -y / np.conj(tap)
        ytf[ell] = -y / tap
        ytt[ell] = y + charging
        ybus[i, i] += yff[ell]
        ybus[i, j] += yft[ell]
        ybus[j, i] += ytf[ell]
        ybus[j, j] += ytt[ell]

    ybus[np.diag_indices(n_bus)] += (bus[:, 4] + 1j * bus[:, 5]) / base_mva
    return ybus, yff, yft, ytf, ytt


def load_case14(case_file: Path = CASE_FILE) -> Case14:
    """Read the local IEEE-14 PGLib ``.m`` file and return the online network."""
    source = Path(case_file).read_text(encoding="utf-8")
    base_match = re.search(r"mpc\.baseMVA\s*=\s*([-+0-9.eE]+)\s*;", source)
    if base_match is None:
        raise ValueError(f"missing mpc.baseMVA in {case_file}")
    base_mva = float(base_match.group(1))
    bus = _numeric_matrix(source, "bus")
    gen = _numeric_matrix(source, "gen")
    gencost = _numeric_matrix(source, "gencost")
    branch = _numeric_matrix(source, "branch")

    expected_ids = np.arange(1, len(bus) + 1)
    if not np.array_equal(bus[:, 0].astype(int), expected_ids):
        raise ValueError("this implementation requires consecutive MATPOWER bus IDs")
    if len(gen) != len(gencost):
        raise ValueError("generator and generator-cost row counts do not match")
    if np.any(gencost[:, 0] != 2) or np.any(gencost[:, 3] != 3):
        raise ValueError("only quadratic polynomial generator costs are supported")

    online_gen = gen[:, 7] > 0
    gen, gencost = gen[online_gen], gencost[online_gen]
    branch = branch[branch[:, 10] > 0]
    if np.any(branch[:, 5] <= 0):
        raise ValueError("all online branches require a positive rateA limit")

    ybus, yff, yft, ytf, ytt = _network_admittance(base_mva, bus, branch)
    case = Case14(base_mva, bus, gen, branch, gencost, ybus, yff, yft, ytf, ytt)
    if len(np.unique(case.generator_buses)) != case.n_gen:
        raise ValueError("one generator per generator bus is required for AC completion")
    return case


if __name__ == "__main__":
    case = load_case14()
    print(
        f"IEEE 14-bus: {case.n_bus} buses, {case.n_gen} generators, "
        f"{len(case.active_generators)} active generators, {len(case.branch)} branches"
    )
