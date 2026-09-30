"""
Test module: check the validity of the material property 
routines, and test the scaling of the material property. 

    This test submodule configure the simulation at the beginning of each
    session, then, construct a series of test that read the database, and 
    check if the **real** data scaled are equal to the array in pdb. 
"""
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from global_variables import _PATH_, _TEST_, _TOL_

from stonedfenicsx.config.input_parser import parse_input
from stonedfenicsx.config.phase_db import read_capacity, read_diffusivity, read_expansivity, read_rheology
from stonedfenicsx.config.simulation_config import configure_simulation

# Generating a class that contains the test, such that I call one time the configuration routine 
# --- 
_SET_SHEAR_ZONE_ = {'shear_heating_disl_law','shear_heating_disl_tau_min','shear_heating_disl_phi'}
_SET_ALPHA_ = {'id_ph','alpha0','name_alpha'}
_SET_RHO_ = {'id_ph','rho0','name_density'}
_SET_K_ = {'id_ph','k','name_conductivity'}
_SET_CP_ = {'id_ph','cp','name_capacity'}
_SET_VIS_ = {'id_ph','name_diffusion','name_dislocation','eta'}
_DICT_ = {'alpha':_SET_ALPHA_
          ,'rho':_SET_RHO_
          ,'cond':_SET_K_
          ,'capc':_SET_CP_
          ,'visc':_SET_VIS_}
# --- 

#@pytest.fixture(scope="session", autouse=True)
def cleanup_output():
    yield
    pt = str(Path(__file__).resolve().parents[0])
    shutil.rmtree(f"{pt}/{_PATH_}", ignore_errors=True)

#@pytest.fixture(scope="session", autouse=True)
def configure() -> dict:
    """Test Function for configuring the simulation
    It serves for debugging purpose and as a stand-alone test.
    It releases the following data: 
    a. pdb, ph_in (input_interface) and sc
    The test are aimed to collect the material property and 
    test if from the scaled value is possible to obtain the 
    original value. 
    
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
    
    _, _, pdb, sc = configure_simulation(ph_in, input_data)
    
    return {'pdb':pdb ,'sc':sc, 'ph_in':asdict(ph_in)}


def iterate_ph_in(ph_in,exclude:bool=True,target:str='alpha')->dict:
    """Iterate in the input database and collect the target property 

    Args:
        ph_in (PHINPUT not imported here): Phase database from the input configuration file (input.yml)
        exclude (bool, optional): flag. Defaults to True. This flag exclude the property of the shear zone
        target (str, optional): target property (i.e., rheology)

    Returns:
        dict: collection of property 
            {phase} : {id: prop associated}
            
    Pseudocode: 
                -> exclude the shear zone data
                -> extract the reference set 
                -> create empty dictionary
                -> loop over the ph_in with **i** as index
                  -> if i not in set_shear: 
                  ->     convert thedataclass into a dictionary
                  ->     create empty dictionary 2
                  ->     loop over the key dict with **k** as index
                          -> if k in reference set 
                          -> fill empty_dictionary_2 with the relative property
                   -> update the dictionary empty_dictionary_1 with empty_dictionary_2 
                -> return empty_dictionary_1 
    """

    reference_set = _DICT_[target]
    # ph_in is already passed through as_dict
    target_array = {} # empty dictionary
    for i in ph_in:
        # Check if the property i not belongs to _SET_SHEAR_ZONE
        if i not in _SET_SHEAR_ZONE_:
            # convert class into a dictionary
            ph = ph_in[i]
            # empty dictionary
            phase_array = {}
            for k in ph:
                if k in reference_set:
                    phase_array[k] = ph[k]
            target_array[i] = phase_array
        else: 
            if not exclude and target == 'Rheology':
                pass 
        
            #target_array[i] = phase_array

    return target_array

# --- 
def test_alpha(configure):
    # Extract relevant data: 
    pdb=configure['pdb']
    ph_in=configure['ph_in']
    sc = configure['sc']
    
    # --- 
    def clausure_alpha_check(target_array:dict,sc,pdb)->bool:
        
        name = target_array['name_alpha']
        alpha0 = target_array['alpha0']
        id_ph = target_array['id_ph'] - 1
        if name in 'Constant':
            approved_test = np.isclose(alpha0
                                       ,pdb.alpha0[id_ph] *  1 / sc.temp
                                       ,_TOL_)
        else: 
            buf = read_expansivity(name)
            a = np.isclose(buf.alpha0
                           ,pdb.alpha0[id_ph] * 1 / sc.temp
                           ,_TOL_)
            b = np.isclose(buf.alpha1
                           ,pdb.alpha1[id_ph] *  1 / sc.temp**2
                           ,_TOL_)
            c = np.isclose(buf.alpha2
                           ,pdb.alpha2[id_ph] *  1 / sc.stress
                           ,_TOL_)
            approved_test =  all([a, b, c])
            
        
        
        return approved_test
    # --- 
    target_alpha_array = iterate_ph_in(ph_in=ph_in
                                       ,exclude=True
                                       ,target='alpha')
    for i in target_alpha_array: 
        approved = clausure_alpha_check(target_array=target_alpha_array[i]
                                        ,pdb=pdb,sc=sc)
        try: 
            assert approved 
        except Warning as w:
            raise Warning(f'{i} for alpha property failed the test',w)
            assert approved

# --- 
def test_cp(configure):
    # Extract relevant data: 
    pdb=configure['pdb']
    ph_in=configure['ph_in']
    sc = configure['sc']
    
    # --- 
    def clausure_cp_check(target_array:dict,sc,pdb)->bool:

        scal_c1 = sc.energy / sc.mass / sc.temp ** (0.5)
        scal_c2 = (sc.energy * sc.temp) / sc.mass
        scal_c3 = (sc.energy * sc.temp**2) / sc.mass
        scal_c4 = (sc.energy) / sc.mass / sc.temp**2
        scal_c5 = (sc.energy) / sc.mass / sc.temp**3

        name = target_array['name_capacity']
        cp = target_array['cp']
        id_ph = target_array['id_ph'] - 1
        if name in 'Constant':
            approved_test = np.isclose(cp
                                       ,pdb.c0[id_ph] *  sc.cp
                                       ,_TOL_)
        else: 
            buf = read_capacity(name)
            a = np.isclose(buf.c0
                          ,pdb.c0[id_ph] *  sc.cp
                          ,_TOL_)
            b = np.isclose(buf.c1
                          ,pdb.c1[id_ph] * scal_c1
                          ,_TOL_)
            c = np.isclose(buf.c2 
                          ,pdb.c2[id_ph] * scal_c2
                          ,_TOL_)
            d = np.isclose(buf.c3
                          ,pdb.c3[id_ph] * scal_c3
                          ,_TOL_)
            e = np.isclose(buf.c4
                          ,pdb.c4[id_ph] * scal_c4
                          ,_TOL_)
            f = np.isclose(buf.c5
                          ,pdb.c5[id_ph] * scal_c5
                          ,_TOL_)            
            approved_test =  all([a, b, c, d, e, f])
            
        
        
        return approved_test
    # --- 
    target_cp_array = iterate_ph_in(ph_in=ph_in
                                       ,exclude=True
                                       ,target='capc')
    for i in target_cp_array: 
        approved = clausure_cp_check(target_array=target_cp_array[i]
                                        ,pdb=pdb,sc=sc)
        try: 
            assert approved 
        except Warning as w:
            raise Warning(f'{i} for capacity property failed the test',w)
            assert approved


#def test_conductivity(configure_simulation):
#def test_rho(configure_simulation):
#


def place_holder_phase_pdb():
    """Place holder -> configure material property, scaling them and read the 
    database to see the scaling if it holds

    Returns:
        _type_: _description_
    """
    # Test rheology
    rqrtz = read_rheology("Wet_Quartzite_2001_Dislocation_creep", 1)
    rolivinedsl = read_rheology("Hirth_wet_Dislocation_creep", 1)
    rolivinedff = read_rheology("Hirth_wet_Diffusion_creep", 0)
    # Test Heat Capacity
    cp0 = read_capacity("Mantle_Bernard_Ar_199x_FA")
    cp1 = read_capacity("Mantle_Bernard_Ar_199x_FO")
    cp2 = read_capacity("Mantle_Bernard_Ar_199x_FO_FA")
    cp3 = read_capacity("Mantle_Bernard_1988_FA")
    cp4 = read_capacity("Mantle_Bernard_1988_FO")
    cp5 = read_capacity("Mantle_Bernard_1988_FO_FA")
    cp6 = read_capacity("Crust")
    # Thermal diffusivity
    dif_0 = read_diffusivity("Mantle_Richards_2018")
    dif_1 = read_diffusivity("Crust_Richards_2018")
    # Thermal expansivity
    alpha_0 = read_expansivity("Mantle")
    alpha_1 = read_expansivity("Oceanic_crust")

    return 0

if __name__ == "__main__":
    #test_alpha(configure=configure())
    test_cp(configure=configure())