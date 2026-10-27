"""
Benchmark of the interface shear heating against Abers, van Keken & Wilson
(2020), Geosphere 16(6), 1408-1424, doi:10.1130/GES02278.1, Figs. 3 and 4A.

Abers et al. (2020) setup (their Fig. 3/4 caption and "Numerical models"):
    planar slab dipping 20 deg, V = 50 mm/yr, incoming plate 50 Ma,
    full decoupling to 80 km, linear taper to full coupling at 82.5 km,
    half-space cooling inflow with Tp = 1350 C, steady state,
    domain 652 x 230 km, dry-olivine wedge, rigid overriding crust,
    ductile shear zone = wet quartzite (Hirth et al., 2001) with the
    Shinevar et al. (2015) fugacity (a_H2O = 1),
    tau = tau_v * tanh(mu' sigma_n / tau_v)  (Gao & Wang, 2017; their eq. 4),
    mu' = 0, 0.03, 0.05, 0.10, 0.15.

Published numbers used as targets (their p. 1414-1415):
    (a) at 20 km depth, mu' 0.03 -> 0.15 raises T_f by 139 C
        (136 C with the analytical approximation);
    (b) at 65 km depth the same mu' range gives a 32 C spread;
    (c) for mu' > 0.05 the slab surface is nearly isothermal from the
        brittle-ductile transition to the decoupling depth, at ~390 C;
    (d) maximum interface temperatures of 350-400 C (Fig. 4 caption).
    All their temperatures include a 0.3 C/km adiabat added a posteriori,
    so the same correction is added here before comparing absolute values
    (it cancels in differences).

Mapping onto StonedFeniCSx:
    mu' = sin(phi), because the code uses tau_lim = P sin(phi) with P the
    lithostatic pressure (Abers: tau_f = mu' sigma_n).
    Viscous stress: the code uses the invariant form with eps_II = V/(2w);
    Abers use tau_v = (Delta sigma)/2 with eps = V/w and the uniaxial
    prefactor. For n = 4 the two differ by a factor 2 * 3^(-(n+1)/(2n))
    = 1.006, i.e. they are the same law.

Differences that remain (they shift absolute temperatures by a few tens of
degrees at most, and are the reason for the tolerances below):
    * inflow: 1D plate model (Crank-Nicolson, slab_tk thick) instead of the
      half-space solution; at 50 Ma the two differ little in the top 30 km;
    * arc-side boundary: linear geotherm reaching Tp at lab_d, with lab_d
      chosen so that the surface heat flow is 65 mW/m^2 (Abers use a
      65 mW/m^2 geotherm with crustal heat production);
    * no radiogenic heating, constant k, rho, cp everywhere;
    * shear-zone width w = 0.5 km (input_tests.yaml); Abers' value is in
      their Table S3 and was not checked here. tau_v ~ w^(-1/4), so a factor
      2 in w changes tau_v by ~16 % and the plateau temperature by ~10 C;
    * the flow law sees T without the adiabat (Abers add it), which makes
      the shear zone here slightly stronger.

The mu' = 0 reference is a "Constant" run with tau = 0 (see the note in
test_shear_heating_verification.py on why NoShear is not used).
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
_FOLDER_ = "ShearHeatingAbers2020"
_SECONDS_PER_YEAR_ = 365.25 * 24 * 3600

# Abers et al. (2020) Fig. 4 configuration
_DIP_ = 20.0  # deg
_V_CM_YR_ = 5.0
_AGE_ = 50.0  # Myr
_TP_ = 1350.0  # C
_DECOUPLING_ = 80.0  # km
_TAPER_ = 2.5  # km, 80 -> 82.5 km
_Q_SURF_ = 65.0e-3  # W/m^2, arc-side surface heat flow
_ADIABAT_ = 0.3  # C/km, added a posteriori as in Abers et al.
_MU_ = (0.03, 0.05, 0.10, 0.15)

# Constant properties
_K_ = 3.0
_RHO_ = 3300.0
_CP_ = 1250.0
_G_ = 9.81

_PHASES_ = (
    "subducting_plate_mantle",
    "oceanic_crust",
    "wedge_mantle",
    "overriding_mantle",
    "overriding_upper_crust",
    "overriding_lower_crust",
)


def _name(mu: float) -> str:
    return f"AB_mu{int(round(mu * 100)):03d}"


def _path_save() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), _FOLDER_)


def _setup(test_name: str, model_shear: str, phi: float = 0.0):
    path_input = f"{Path(__file__).resolve().parents[0]}/input_tests.yaml"
    inp, ph_input = parse_input(path_input)

    # Geometry: straight 20 deg slab, 230 km deep domain (Abers: 652 x 230 km)
    lab_d = _K_ * _TP_ / _Q_SURF_ * 1e-3  # km, linear geotherm with q_s = 65 mW/m^2
    inp.g_input.van_keken = False
    inp.g_input.slab_type = "CustomRibe"
    inp.g_input.sub_constant_flag = True
    inp.g_input.sub_theta_max = _DIP_  # degrees, converted inside stoned_fenicsx
    inp.g_input.x = np.array([0.0, 660.0])
    inp.g_input.y = np.array([-230.0, 0.0])
    inp.g_input.cr = 0.0
    inp.g_input.lc = 0.0
    inp.g_input.ocr = 6.0
    inp.g_input.lab_d = lab_d
    inp.g_input.lit_mt = lab_d
    inp.g_input.decoupling = _DECOUPLING_
    inp.g_input.transition = _TAPER_
    inp.g_input.resolution_refine = 1.0  # Abers: 1 km along the slab surface
    inp.g_input.redo_mesh = True

    # Thermal and kinematic boundary conditions
    inp.ctrl_tbc.temp_max = _TP_
    inp.ctrl_tbc.slab_age = _AGE_
    inp.ctrl_tbc.right_boundary = "Continental"
    inp.ctrl_ky.v_s = np.array([_V_CM_YR_, 0.0], dtype=np.float64)

    # Steady state, same tolerances as the paper runs
    inp.ctrl.steady_state = 1
    inp.ctrl.decoupling_ctrl = 1
    inp.ctrl.tol = 1e-6
    inp.ctrl.it_max = 50
    inp.ctrl.tol_dtemp = 1e-2
    inp.ctrl.model_shear = model_shear

    ph_input.shear_heating_disl_law = "Wet_Quartzite_2001_Dislocation_creep"
    ph_input.shear_heating_disl_phi = phi
    ph_input.shear_heating_disl_tau_min = 0.0

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
        ph.rho0 = _RHO_
        ph.k = _K_
        ph.cp = _CP_
    # Dry-olivine wedge, as in Abers et al. (2020) and van Keken et al. (2008)
    ph_input.wedge_mantle.name_diffusion = "VK_Diffusion_creep"
    ph_input.wedge_mantle.name_dislocation = "VK_Dislocation_creep"

    inp.ctrl_io.test_name = test_name
    inp.ctrl_io.path_save = _path_save()
    return inp, ph_input


def _slab_surface_temperature(test_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Depth [km, positive down] and temperature [C] along the slab surface (tags 8, 9)."""
    with h5py.File(os.path.join(_path_save(), test_name, "Steady_state.h5"), "r") as fl:
        xy = np.array(fl["/Mesh/mesh/geometry"])
        temp = np.array(fl["/Function/Temperature  [degC]/0"]).flatten()
        tag = np.array(fl["/Function/MeshTAG/0"]).flatten()
    on_slab = (tag == 8) | (tag == 9)
    depth = -xy[on_slab, 1]
    order = np.argsort(depth)
    depth, idx = np.unique(depth[order], return_index=True)
    return depth, temp[on_slab][order][idx]


def _temperature_at(test_name: str, depths: np.ndarray, adiabat: bool = False) -> np.ndarray:
    z, t = _slab_surface_temperature(test_name)
    t_z = np.interp(depths, z, t)
    return t_z + _ADIABAT_ * depths if adiabat else t_z


@pytest.fixture(scope="module")
def runs():
    """Run every case once; tests below only read the outputs."""
    cases = {_name(0.0): ("Constant", 0.0)}
    cases.update({_name(mu): ("SelfConsistent", float(np.degrees(np.arcsin(mu)))) for mu in _MU_})
    for name, (mode, phi) in cases.items():
        inp, ph_input = _setup(name, mode, phi)
        stoned_fenicsx(inp=inp, ph_in=ph_input)
    yield
    if not DEBUG and MPI.COMM_WORLD.rank == 0:
        shutil.rmtree(_path_save(), ignore_errors=True)


def _frictional_dt_analytical(dmu: float, z_km: float) -> float:
    """Molnar & England (1990) / van Keken et al. (2019) approximation for the
    temperature rise from a frictional stress tau = mu' rho g z on a planar
    fault (Abers et al., 2020, eq. 5, shear-heating term):
        dT = (z / k) * V * dmu' * rho g z / S2,
        S2 = 1 + (3 sqrt(pi) / 4) * sqrt(V z sin(dip) / kappa)."""
    v = _V_CM_YR_ * 1e-2 / _SECONDS_PER_YEAR_
    z = z_km * 1e3
    kappa = _K_ / (_RHO_ * _CP_)
    s2 = 1.0 + 0.75 * np.sqrt(np.pi) * np.sqrt(v * z * np.sin(np.radians(_DIP_)) / kappa)
    return (z / _K_) * v * dmu * _RHO_ * _G_ * z / s2


def test_brittle_regime_friction_sensitivity(runs):
    """(a) mu' 0.03 -> 0.15 at 20 km: Abers 139 C, analytical 136 C."""
    if MPI.COMM_WORLD.rank != 0:
        return
    z = np.array([20.0])
    dt_num = float(_temperature_at(_name(0.15), z)[0] - _temperature_at(_name(0.03), z)[0])
    dt_ana = _frictional_dt_analytical(0.12, 20.0)
    print(f"dT(20 km, mu' 0.03 -> 0.15): StonedFeniCSx {dt_num:.1f} C, "
          f"analytical (this setup) {dt_ana:.1f} C, Abers numerical 139 C")
    assert np.isclose(dt_num, dt_ana, rtol=0.15, atol=0.0)
    assert np.isclose(dt_num, 139.0, rtol=0.20, atol=0.0)


def test_ductile_regime_buffering(runs):
    """(b) at 65 km the mu' 0.03-0.15 spread is small (Abers: 32 C) and much
    smaller than at 20 km."""
    if MPI.COMM_WORLD.rank != 0:
        return
    t20 = np.array([_temperature_at(_name(mu), np.array([20.0]))[0] for mu in _MU_])
    t65 = np.array([_temperature_at(_name(mu), np.array([65.0]))[0] for mu in _MU_])
    spread20, spread65 = np.ptp(t20), np.ptp(t65)
    print(f"T(65 km) for mu' {_MU_}: {np.round(t65, 1)} C\n"
          f"spread 20 km {spread20:.1f} C, spread 65 km {spread65:.1f} C (Abers 32 C)")
    assert spread65 < 60.0
    assert spread65 < 0.4 * spread20


def test_isothermal_plateau(runs):
    """(c) for mu' >= 0.10 the slab surface between 45 and 70 km is nearly
    isothermal at ~390 C (adiabat added as in Abers et al.).

    The window stops at 70 km, i.e. 10 km above the decoupling depth. In the
    last few km above it the slab surface heats up quickly by conduction from
    the hot wedge corner, which is not part of the plateau. The code's tanh
    taper is also centred at 80 km (f = 0.5 there), whereas Abers start their
    linear taper at 80 km, so the wedge flow starts ~1 km shallower here."""
    if MPI.COMM_WORLD.rank != 0:
        return
    z = np.linspace(45.0, 70.0, 6)
    temps = {mu: _temperature_at(_name(mu), z, adiabat=True) for mu in (0.10, 0.15)}
    for mu, t in temps.items():
        print(f"mu' = {mu:.2f}: T(45-70 km) = {np.round(t, 1)} C, mean {t.mean():.1f} C, range {np.ptp(t):.1f} C")
    for t in temps.values():
        assert 350.0 < t.mean() < 430.0
        assert np.ptp(t) < 50.0


def test_maximum_temperature_above_decoupling(runs):
    """(d) between 10 and 70 km the interface stays below ~420 C for every mu'
    (Abers: maximum 350-400 C), and shear heating always warms it relative
    to mu' = 0. The window stops 10 km above decoupling for the reason given
    in test_isothermal_plateau."""
    if MPI.COMM_WORLD.rank != 0:
        return
    z = np.linspace(10.0, 70.0, 31)
    t0 = _temperature_at(_name(0.0), z, adiabat=True)
    temps = {mu: _temperature_at(_name(mu), z, adiabat=True) for mu in _MU_}
    for mu, t in temps.items():
        print(f"mu' = {mu:.2f}: max T(10-70 km) = {t.max():.1f} C")
    for t in temps.values():
        assert t.max() < 420.0
        assert np.all(t >= t0 - 0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])