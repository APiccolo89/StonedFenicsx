"""
Verification tests for the interface shear heating.

All runs use the van Keken et al. (2008) planar geometry (45 deg straight slab,
50 Myr plate, 5 cm/yr), constant thermal properties and a constant-viscosity
wedge, so that the energy equation is linear whenever the shear-heating source
does not depend on temperature. The decoupling depth is 80 km; the checks are
made on the slab surface between 10 and 40 km depth, i.e. below a rigid
overriding lid and well above decoupling.

The reference run is a "Constant" run with tau = 0, not a "NoShear" run: a
fully linear NoShear configuration stops the outer loop after two relaxed
iterations and is not fully converged.

Three checks:

1. Superposition (exact). In "Constant" mode the source tau*v is independent of
   T, so the temperature rise must scale exactly with tau:
       T(2 tau) - T(0) = 2 [T(tau) - T(0)].
   Tests that the source enters the energy equation as a fixed, correctly
   assembled term.

2. Magnitude against the Molnar & England (1990) solution. For a constant shear
   stress on a planar fault under a conducting half-space, the shear-heating
   contribution to the fault temperature is approximately
       dT(z) = tau * V * z / (k * S(z)),   S(z) = 1 + b * sqrt(V z sin(delta) / kappa),
   with b = sqrt(pi)/2 for a constant heat flux into the fault (van Keken et al.,
   2019, G-cubed, doi:10.1029/2019GC008334). The approximation is derived for
   shallow dips; at 45 deg it is only accurate to roughly 10-20 %, so the
   tolerance is loose. A failure well outside it points to a wrong prefactor
   (units, facet measure, decoupling taper); a marginal failure should be
   checked against the exact expressions in van Keken et al. (2019).

3. Friction-angle sensitivity ("SelfConsistent", wet quartzite). Where the
   interface is frictional, tau_eff = P sin(phi) does not depend on T, so the
   shear-heating temperature rise must scale with sin(phi). At 20 km depth the
   corrected flow law keeps the interface frictional, so the ratio of the
   temperature rises for phi = 5 and 2.5 deg must be close to
   sin(5)/sin(2.5) ~= 2. With the former MPa/Pa error the interface was viscous
   there and the ratio was close to 1.
"""

import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
from mpi4py import MPI

from stonedfenicsx.config.input_parser import parse_input
from stonedfenicsx.stoned_fenicsx import stoned_fenicsx

DEBUG = False  # keep the output folder for inspection
_FOLDER_ = "ShearHeatingVerification"
_TAU_ = 20.0e6  # constant shear stress [Pa]
_DEPTHS_ = np.array([10.0, 20.0, 30.0, 40.0])  # slab-surface depths checked [km]
_SECONDS_PER_YEAR_ = 365.25 * 24 * 3600
_PHASES_ = (
    "subducting_plate_mantle",
    "oceanic_crust",
    "wedge_mantle",
    "overriding_mantle",
    "overriding_upper_crust",
    "overriding_lower_crust",
)


def _path_save() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), _FOLDER_)


def _setup(test_name: str, model_shear: str, tau: float = 0.0, phi: float = 0.0):
    """Planar van Keken geometry, constant properties, linear wedge."""
    path_input = f"{Path(__file__).resolve().parents[0]}/input_tests.yaml"
    inp, ph_input = parse_input(path_input)

    inp.g_input.cr = 0.0
    inp.g_input.lc = 0.0
    inp.g_input.ocr = 6.0
    inp.g_input.lit_mt = 50.0
    inp.g_input.lab_d = 50.0
    inp.g_input.decoupling = 80.0
    inp.g_input.van_keken = True

    inp.ctrl.steady_state = 1
    inp.ctrl.decoupling_ctrl = 1
    inp.ctrl.model_shear = model_shear

    ph_input.shear_heating_disl_tau_min = tau
    ph_input.shear_heating_disl_phi = phi
    ph_input.shear_heating_disl_law = "Wet_Quartzite_2001_Dislocation_creep"

    for name in _PHASES_:
        ph = getattr(ph_input, name)
        ph.name_capacity = "Constant"
        ph.name_conductivity = "Constant"
        ph.name_alpha = "Constant"
        ph.name_density = "Constant"
        ph.name_diffusion = "Constant"
        ph.name_dislocation = "Constant"
        ph.radiative_conductivity = 0
        ph.radiogenic_heat = 0.0
        ph.rho0 = 3300.0
        ph.k = 3.0
        ph.cp = 1250.0

    inp.ctrl_io.test_name = test_name
    inp.ctrl_io.path_save = _path_save()
    return inp, ph_input


def _slab_surface_temperature(test_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Depth [km, positive down] and temperature [degC] along the slab surface (tags 8, 9)."""
    with h5py.File(os.path.join(_path_save(), test_name, "Steady_state.h5"), "r") as fl:
        xy = np.array(fl["/Mesh/mesh/geometry"])
        temp = np.array(fl["/Function/Temperature  [degC]/0"]).flatten()
        tag = np.array(fl["/Function/MeshTAG/0"]).flatten()
    on_slab = (tag == 8) | (tag == 9)
    depth = -xy[on_slab, 1]
    order = np.argsort(depth)
    depth, idx = np.unique(depth[order], return_index=True)
    return depth, temp[on_slab][order][idx]


def _temperature_at(test_name: str, depths: np.ndarray) -> np.ndarray:
    z, t = _slab_surface_temperature(test_name)
    return np.interp(depths, z, t)


@pytest.fixture(scope="module")
def runs():
    """Run every case once; tests below only read the outputs."""
    cases = {
        "SH_tau0": ("Constant", 0.0, 0.0),
        "SH_tau": ("Constant", _TAU_, 0.0),
        "SH_2tau": ("Constant", 2.0 * _TAU_, 0.0),
        "SH_phi2p5": ("SelfConsistent", 0.0, 2.5),
        "SH_phi5": ("SelfConsistent", 0.0, 5.0),
    }
    meta = {}
    for name, (mode, tau, phi) in cases.items():
        inp, ph_input = _setup(name, mode, tau, phi)
        if name == "SH_tau":  # read before the run, the configuration scales them in place
            meta["v"] = float(inp.ctrl_ky.v_s[0]) * 1e-2 / _SECONDS_PER_YEAR_  # m/s
            meta["k"] = ph_input.subducting_plate_mantle.k
            meta["kappa"] = ph_input.subducting_plate_mantle.k / (
                ph_input.subducting_plate_mantle.rho0 * ph_input.subducting_plate_mantle.cp
            )
        stoned_fenicsx(inp=inp, ph_in=ph_input)
    yield meta
    if not DEBUG and MPI.COMM_WORLD.rank == 0:
        shutil.rmtree(_path_save(), ignore_errors=True)


def test_superposition_constant_stress(runs):
    if MPI.COMM_WORLD.rank != 0:
        return
    t0 = _temperature_at("SH_tau0", _DEPTHS_)
    d1 = _temperature_at("SH_tau", _DEPTHS_) - t0
    d2 = _temperature_at("SH_2tau", _DEPTHS_) - t0
    print(f"dT(tau)  = {np.round(d1, 2)} K\ndT(2tau) = {np.round(d2, 2)} K")
    assert np.all(d1 > 0.0)
    assert np.allclose(d2, 2.0 * d1, rtol=1e-2, atol=0.5)


def test_constant_stress_against_molnar_england(runs):
    if MPI.COMM_WORLD.rank != 0:
        return
    v, k, kappa = runs["v"], runs["k"], runs["kappa"]
    dip = np.radians(45.0)
    z = _DEPTHS_ * 1e3
    s = 1.0 + 0.5 * np.sqrt(np.pi) * np.sqrt(v * z * np.sin(dip) / kappa)
    d_ref = _TAU_ * v * z / (k * s)
    d_num = _temperature_at("SH_tau", _DEPTHS_) - _temperature_at("SH_tau0", _DEPTHS_)
    print(f"z = {_DEPTHS_} km\nnumerical  dT = {np.round(d_num, 1)} K\nanalytical dT = {np.round(d_ref, 1)} K")
    assert np.allclose(d_num, d_ref, rtol=0.25)


def test_friction_angle_scaling(runs):
    if MPI.COMM_WORLD.rank != 0:
        return
    depth = np.array([20.0])
    t0 = _temperature_at("SH_tau0", depth)
    d_small = _temperature_at("SH_phi2p5", depth) - t0
    d_large = _temperature_at("SH_phi5", depth) - t0
    ratio = float(d_large[0] / d_small[0])
    print(f"dT(phi=2.5) = {d_small[0]:.1f} K, dT(phi=5) = {d_large[0]:.1f} K, ratio = {ratio:.2f}")
    assert 1.6 < ratio < 2.1


if __name__ == "__main__":
    pytest.main([__file__, "-s"])