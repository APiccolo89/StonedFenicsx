"""Test configuration module

The configuration tests simply checks wheter or not the mesh is created, or if the scaling
are read properly.

This section is still under-construction, and most likely will results in a few refractoring.
Each part of the configuration module is interconnected, the next steps are to create indipendency between
the component and testing if from the scaled material parameters is possible to recompute the original
values.
"""

import shutil
from pathlib import Path

import dolfinx
import numpy as np
import pytest

from stonedfenicsx.config.input_parser import parse_input
from stonedfenicsx.config.simulation_config import configure_simulation

from .global_variables import _PATH_, _TEST_


@pytest.fixture(scope="session", autouse=True)
def cleanup_output():
    yield
    pt = str(Path(__file__).resolve().parents[0])
    shutil.rmtree(f"{pt}/{_PATH_}", ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def configure() -> int:
    """Test Function for configuring the simulation
    It serves for debugging purpose and as a stand-alone test.
    """

    # Find the main folder of the package
    pkg_root = Path(__file__)
    # Select the appropriate path for the input file
    input_file = Path(pkg_root.parents[0], "input_tests.yaml")
    # parse the input file
    input_data, ph_in = parse_input(input_file)
    # Set the path of the tests
    path_save = pkg_root.parents[0] / _PATH_
    test_name = _TEST_
    input_data.ctrl_io.test_name = test_name
    input_data.ctrl_io.path_save = path_save

    ph_in.oceanic_crust.name_alpha = "Oceanic_crust"
    ph_in.oceanic_crust.name_capacity = "Oceanic_crust"
    ph_in.oceanic_crust.radiative_conductivity = 1
    ph_in.oceanic_crust.rho0 = 2800
    ph_in.oceanic_crust.name_conductivity = "Crust_Richards_2018"
    ph_in.oceanic_crust.name_density = "PT"

    ph_in.subducting_plate_mantle.name_capacity = "Mantle_Bernard_Ar_199x_FO_FA"
    ph_in.subducting_plate_mantle.name_conductivity = "Mantle_Richards_2018"
    ph_in.subducting_plate_mantle.name_alpha = "Mantle"
    ph_in.subducting_plate_mantle.rho0 = 3300
    ph_in.subducting_plate_mantle.name_density = "PT"

    ph_in.wedge_mantle.name_capacity = "Mantle_Bernard_Ar_199x_FO_FA"
    ph_in.wedge_mantle.name_conductivity = "Mantle_Richards_2018"
    ph_in.wedge_mantle.name_alpha = "Mantle"
    ph_in.wedge_mantle.rho0 = 3300
    ph_in.wedge_mantle.name_density = "PT"
    ph_in.wedge_mantle.name_dislocation = "VK_Dislocation_creep"
    ph_in.wedge_mantle.name_diffusion = "VK_Diffusion_creep"
    ctrl_sim, mesh, pdb, sc = configure_simulation(ph_in, input_data)

    return {"ctrl_sim": ctrl_sim, "mesh": mesh, "pdb": pdb, "sc": sc}


def test_scaling(configure):
    """
    Test the scaling from input
    """
    sc = configure["sc"]
    assert sc.length == 600e3
    assert sc.eta == 1e21
    assert sc.temp == 1333.0
    assert sc.stress == 1e9


def test_output(configure):
    """Test if all the folder have been created"""
    ctrl_sim = configure["ctrl_sim"]

    assert ctrl_sim.ctrl_io.path_save.is_dir()
    assert ctrl_sim.ctrl_io.path_test.is_dir()
    assert ctrl_sim.ctrl_io.path_cached_information.is_dir()


def test_mesh(configure):

    from stonedfenicsx.create_mesh.aux_create_mesh import dict_tag_lines

    mesh = configure["mesh"]
    sc = configure["sc"]

    # Do The domain exists?
    assert hasattr(mesh, "global_domain")
    assert hasattr(mesh, "wedge_domain")
    assert hasattr(mesh, "subduction_plate_domain")
    assert hasattr(mesh, "crust_domain")

    # Do the basic mesh complies to the internal rule?
    lower_left_corner = np.array(
        [np.min(mesh.global_domain.mesh.geometry.x[:, 0]), np.min(mesh.global_domain.mesh.geometry.x[:, 1])]
    )
    lower_left_corner = lower_left_corner * sc.length
    lower_right_corner = np.array(
        [np.max(mesh.global_domain.mesh.geometry.x[:, 0]), np.min(mesh.global_domain.mesh.geometry.x[:, 1])]
    )
    lower_right_corner = lower_right_corner * sc.length

    # Check left lower corner
    assert np.isclose(lower_left_corner[0], 0, 1e-2)
    assert np.isclose(lower_left_corner[1], -600e3, 1e-2)
    # Check right lower corner
    assert np.isclose(lower_right_corner[0], 660e3, 1e-2)
    assert np.isclose(lower_right_corner[1], -600e3, 1e-2)
    # Check if the slab point is at 60 km from the lower left corner
    facets = mesh.global_domain.facets.find(dict_tag_lines["Subduction_top_wed"])
    assert facets.size != 0
    # (n_facets, n_nodes_per_facet) array of geometry dof indices
    geom = dolfinx.mesh.entities_to_geometry(mesh.global_domain.mesh, 1, facets)
    # Check for duplicate and purge them
    nodes = np.unique(geom.reshape(-1))
    min_slab_x = np.max(mesh.global_domain.mesh.geometry.x[nodes, 0]) * sc.length
    assert np.isclose(660e3 - min_slab_x, 60e3, 1e-2)
