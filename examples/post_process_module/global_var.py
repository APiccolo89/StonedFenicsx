"""
Global variables that all the scripts must know 
"""
_TP_ = 1350.0
_DECOUPLING_ = 80.0 # The default decoupling is 80 km
_PHI_ = 5.0 # The default phi is 5
_SHEAR_HEATING_LAW = "Wet_Quartzite_2001_Dislocation_creep"
_NAME_DIFFUSION_CREEP_ = "VK_Diffusion_creep"
_NAME_DISLOCATION_CREEP_ = "VK_Dislocation_creep"
_K_ = 273.15 # Should I really give you the name? 
_PA_ = 1e9 # The data are always in GPa, so this is the conversion
dict_td = {"1": "SS", "0": "TD"}
_R_GRID_ = 1600 # Resolution of the regular grid
_T_CUT_OFF_LIT = 1250 # Cut off for defining the lithosphere
_V_CONVERSION_KM_MYR_ = 1e-5 * 1e6 # velocity conversion into km/Myr
_POINT_FILTER_OVERRIDING = -300.0 # This point depends on the max coordinate of x so there is only the depth