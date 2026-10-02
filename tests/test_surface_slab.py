from importlib.resources import files
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from stonedfenicsx.config.geometry import GeomInput
from stonedfenicsx.create_mesh.aux_create_mesh import function_create_subducting_plate_geometry

from .global_variables import _TOL_

_DICT_CONFIGURATION_ = {
    "ribe_geometry": {"slab_type": "CustomRibe", "sub_theta_max": 45.0, "sub_lb": 200.0},
    "parabolic_geometry": {"slab_type": "CustomParabolic", "sub_parabolic_a": 5e-4},
    "japan": {"slab_type": "FromFile", "sub_path": "Japan"},
    "mexico": {"slab_type": "FromFile", "sub_path": "Mexico"},
    "chile": {"slab_type": "FromFile", "sub_path": "Chile"},
    "tonga": {"slab_type": "FromFile", "sub_path": "Tonga"},
}
# ---
# ---
def define_geometrical_input(dict_configuration: dict) -> GeomInput:
    """Generate an instance of geometrical input using the
    configuration dictionary

    Args:
        dict_configuration (dict): arbitrary dictionary that
        configure the class geometrical input for the given
        test

    Returns:
        GeomInput: Return the configured geometrical input
        for testing the geometry of the slab.
    """
    # Instance of the class
    g_input = GeomInput()
    # Loop over the configuration dictionary
    for i, v in dict_configuration.items():
        if i == "sub_path":
            # Fetch the path of the current examples:
            # This require the import of `importlib.resources
            # import files`. If someone wants to use the tests
            # as blue print for fetching the real examples
            v = str(
                files("stonedfenicsx").resolve().parents[1]
                / "examples"
                / "data"
                / f"{dict_configuration['sub_path']}_slab.pz"
            )
        setattr(g_input, i, v)
    # Update the class, i.e., check if there are terribly wrong
    # value and convert the angle from degree to radians.
    g_input.van_keken = False
    g_input.check_class_consistency()

    return g_input
# ---
def check_point(tested: NDArray, control: NDArray) -> tuple[bool, int]:
    """Check if the tested array has the same
    number of point of the array of the data
    base

    Args:
        tested (NDArray): current array
        control (NDArray): data base array

    Returns:
        bool: seal of approval
        int : difference of points
    """

    a = len(tested)
    b = len(control)

    return a == b


# ---
def check_length(tested_x: NDArray, tested_y: NDArray, control_x: NDArray, control_y: NDArray) -> tuple[bool, float]:
    """Compute the arc-length of the tested array and control-one
    Check if the cumulative length of the two arrays yield the same
    results.

    Args:
        tested_x (NDArray): tested array coord-x
        tested_y (NDArray): tested array coord-y
        control_x (NDArray): control array coord-x
        control_y (NDArray): control array coord-y

    Returns:
        tuple[bool,float]: tuple: if the length is the same, and the difference of length
    """
    # ---
    def compute_length(a, b):
        return np.max(np.cumsum(np.diff(a) ** 2 + np.diff(b) ** 2))

    return (
        np.isclose(compute_length(tested_x, tested_y), compute_length(control_x, control_y), rtol=_TOL_),
        compute_length(tested_x, tested_y) - compute_length(control_x, control_y),
    )
# ---
def check_coordinate(tx: NDArray, ty: NDArray, cx: NDArray, cy: NDArray) -> bool:
    """_summary_

    Args:
        tx (NDArray): target coord-x
        ty (NDArray): target coord-y
        cx (NDArray): control coord-x
        cy (NDArray): control coord-y

    Returns:
        bool: seal of approval
    """

    seal_approval = True

    count = 0

    for i in range(len(tx)):
        a = np.isclose(tx[i], cx[i], rtol=_TOL_)
        b = np.isclose(ty[i], cy[i], rtol=_TOL_)
        if not a or not b:
            count = count + 1

    if count > 5:
        seal_approval = False

    return seal_approval
# ---
def main_test(g_input: GeomInput, name: str):

    # Define the path where to find the fixed example to compare with
    pt_data = Path(__file__).resolve().parents[0] / "slab_surface"
    pt_data.mkdir(parents=True, exist_ok=True)
    pt_data = pt_data / f"{name}.npz"

    # Create the subducting plate main geometrical point and lines using the information of g_input
    slab_x, slab_y, bot_x, bot_y, oc_cx, oc_cy, g_input = function_create_subducting_plate_geometry(g_input)
    # Load control files
    test_file = np.load(pt_data)

    number_point_slab = all([check_point(slab_x, test_file["slab_x"]), check_point(slab_y, test_file["slab_y"])])
    number_point_ocean = all([check_point(oc_cx, test_file["oc_cx"]), check_point(oc_cy, test_file["oc_cy"])])
    number_point_bottom = all([check_point(bot_x, test_file["bot_x"]), check_point(bot_y, test_file["bot_y"])])

    basic = all([number_point_slab, number_point_ocean, number_point_bottom])
    # Check the lenght of the vectors
    try:
        assert basic
    except AssertionError as w:
        raise AssertionError(
            f"{name} is failing. The database has a different number of point/n",
            f"       Failed: slab = {number_point_slab}, ocean = {number_point_ocean}, bot = {number_point_bottom}",
            w,
        )

    length_s, d_ells = check_length(slab_x, slab_y, test_file["slab_x"], test_file["slab_y"])
    length_o, d_ello = check_length(oc_cx, oc_cy, test_file["oc_cx"], test_file["oc_cy"])
    length_b, d_ellb = check_length(bot_x, bot_y, test_file["bot_x"], test_file["bot_y"])

    length_test = all([length_s, length_o, length_b])

    try:
        assert length_test
    except AssertionError as w:
        raise AssertionError(
            f"{name} is failing. The arrays in the database have a different length/n",
            f"       difference length: slab = {d_ells}, ocean = {d_ello}, bot = {d_ellb}",
            w,
        )

    coord_s = check_coordinate(slab_x, slab_y, test_file["slab_x"], test_file["slab_y"])
    coord_o = check_coordinate(slab_x, slab_y, test_file["slab_x"], test_file["slab_y"])
    coord_b = check_coordinate(slab_x, slab_y, test_file["slab_x"], test_file["slab_y"])

    coord_check = all([coord_s, coord_o, coord_b])

    try:
        assert coord_check
    except AssertionError as w:
        raise AssertionError(
            f"{name} is failing. The arrays in the database have a different length/n",
            f"       Failed: slab = {coord_s}, ocean = {coord_o}, bot = {coord_b}",
            w,
        )
    print("All the test have a seal of approval!")
# ---
def test_ribe_geometry() -> None:
    main_test(define_geometrical_input(_DICT_CONFIGURATION_["ribe_geometry"]), "ribe")
# ---
def test_parabolic_geometry() -> None:
    main_test(define_geometrical_input(_DICT_CONFIGURATION_["parabolic_geometry"]), "parabolic")
# ---
def test_japan_geometry() -> None:
    main_test(define_geometrical_input(_DICT_CONFIGURATION_["japan"]), "japan")
# ---
def test_mexico_geometry() -> None:
    main_test(define_geometrical_input(_DICT_CONFIGURATION_["mexico"]), "mexico")
# ---
def test_chile_geometry() -> None:
    main_test(define_geometrical_input(_DICT_CONFIGURATION_["chile"]), "chile")
# ---
def test_tonga_geometry() -> None:
    main_test(define_geometrical_input(_DICT_CONFIGURATION_["tonga"]), "tonga")
# ---


# ---
# ---
# ---
# TO DO LIST
# [x] Check ribe geometry
# [x] Check parabolic geometry
# [x] Check realistic geometries
# [-] Check boundaries of the final mesh (?)
# workflow: produce slab surface -> save .pz with x,z,ell,theta -> compare with this file => success
#                   oceanic crust -> save.pz file
#
# parabolic geometry => introduce analytical formulation and check directly with that
# Test routine design:
# -----------------------------
# f : Fixature -> [g_input] -> function that receive the data of each test {global data on top of the file i.e. dictionary}
# f : Main test -> use the configured g_input to run the specific test
#                 a. call the function that produce slab top surface [Target routine]
#                 b. call the function that produce crust and bottom of the slab [Target routine]
#                 c. call -> readDB and compare => assert if sx,sy,ell,theta similar to database => if not, throw an error
#                 d. validate geometry : use the length of the slab throw random number (fake velocity,time)=>generate random point
#                    d1: use function to compute the perpendicular point =
#                    d2: check if the resultant array is similar to the bx,by/ox,oy [To do in the future]
#                ======> Print passing test
# f: test_ribe(configure({dict_option}))
# .    main_test(g_input) => so forth
# -----------------------------
# ---
# ---
# def test_real_examples() -> None:
#    set_real_base = {"japan", "mexico", "chile", "tonga"}
#    # loop over and repeat the test
#    for i in set_real_base:
#        main_test(define_geometrical_input(_DICT_CONFIGURATION_[i]), i)
