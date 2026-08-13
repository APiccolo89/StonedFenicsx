# Material properties

Material properties are defined using the input values defined in the *input.yml* (*Material properties* in *How to use*). **StonedFEniCSx** uses the option listed defined in the *input.yml*, over-writes the options that are not needed (see below), and create a small database. The small database is a small collection of arrays associated to specific properties, featuring a size equal to the total number of phases. Internally, **StonedFEniCSx** access to the specific property using the *ID* number of the subregions. 

Inside *config* folder (`\stonedfenicsx\config`) there is a folder containing the material properties and the relative dictionaries. These databases contain the original value of the material properties parameters; these parameters are always converted into the suitable unit of measure (e.g., MPa -> Pa) and then they are divided by the characteristic scales. This process is always done during the configuration stage of the numerical simulation. 

## Rock phase and IDs

The numerical domain is divided into three different computational meshes: overriding plate, subducting plate, and mantle wedge. These subdomains can be made of one or more rocks phase. The rocks phases represent different lithologies, and their ID number connects them to a material database. The mandatory phases are: 

- `subducting plate mantle` (ID = 1): the subducting plate materials.
- `wedge mantle` (ID = 3): convective mantle.
- `overriding plate mantle` (ID = 4): the overriding mantle lithosphere.

The user can introduce crustal levels:

- `oceanic crust` (ID = 2): oceanic crust of the subducting plate.
- `overriding upper crust` (ID = 5): an upper crust layer for the overriding plate.
- `overriding lower crust` (ID = 6): a lower crust layer for the overriding plate.

The customization of the material properties of each phase depends on the subdomain to which these phases belong. For example, `oceanic crust` and `subducting plate mantle` have always a constant viscosity, but they can have different thermal material properties (see **Tab.** {ref}`table:material_phases`). This entails that within the subduction plate, the user choices are overwritten by the default viscosity. This choice design is for extending the code to additional purpose in the future. 


(table:material_phases)=
| Phase name | Optional | Rheology {math}`\eta`| Thermal conductivity {math}`k`| Density {math}`\rho`| Heat capacity {math}`C_p`| Thermal expansion {math}`\alpha` | IDs |
|------------|----------|----------|----------------------|---------|---------------|---------------------|---|
| Subducting plate mantle | No | Constant viscosity | Linear / non-linear| Linear / non-linear | Linear / non-linear | Linear / non-linear | 1|
| Oceanic crust | Yes | Constant viscosity | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear |2|
| Overriding plate mantle | No | Constant viscosity | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear | 3|
| Wedge mantle | No | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear |4|
| Overriding upper crust | Yes | Constant viscosity | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear |5|
| Overriding lower crust | Yes | Constant viscosity | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear|6|

## Material properties

### Rheological material properties

Viscosity can be either constant, temperature dependent or non-linear temperature dependent. The only two phases that can access to different rheological models are:`wedge mantle`. The temperature dependent viscosity is described by the diffusion creep mechanism, while the non-linear temperature dependent viscosity is described by the dislocation creep mechanism. The general equation for both of the mechanisms is:   
```{math}
:label: eq:diffusion_dislocation_creep
\eta_{\mathrm{dif|dis}} =
B_{\mathrm{dif|dis}}
\, \dot{\varepsilon}_{II}^{\,1-\frac{1}{n}}
\exp\!\left(
-\frac{E_{\mathrm{dif|dis}} + P V_{\mathrm{dif|dis}}}{n R T}
\right)
```
{math}`B_{dif|dis}` is the pre-exponential factor for either diffusion (dif) or dislocation creep. {math}`\dot{\varepsilon}_{II}` is the second invariant of the strain rate tensor. {math}`n` is the stress-exponent ({math}`n = 1` in case of diffusion creep mechanism). {math}`E_{\mathrm{dif|dis}}` and {math}`V_{\mathrm{dif|dis}}`. {math}`T` and {math}`P` are the temperature and pressure and {math}`R` is the perfect gas constant. 

User can customize each of the parameter of diffusion and dislocation creep. In **StonedFEniCSx** there is an internal database that collects the rheologies. A small portion of the database is shown to illustrate the diffusion and dislocation creep database:


**diffusion**
```
Common: 
  n: 1.0
  m : 0.0
  d : 1.0
  ah2o: 1.0
  bh2o: 5521e6 
  eh2o: 31.28e3 
  vh2o: -2.009e-5 


Diffusion_creep: 
  Diffusion_DryOlivine: 
    b: 1.5e9
    e: 375.0e3
    v: 5e-6
    f: 'Simpleshear'
    d: 10e3
    mpa: 1
    r: 0
    m: 3.0
    b_si: 'MPa^-1s^-1 m^{m}'
    water_correction: 'None'
    ref: 'Hirth, Greg, and David Kohlstedt. "Rheology of the upper mantle and the mantle wedge: A view from the experimentalists." Geophysical monograph series 138 (2003): 83-105.'
```
This database is build with the original rheological data. There are a few common parameters (e.g., the water fugacity parameters). 
- b: pre-exponential factor 
- e: activation energy
- v: activation volume
- f: Correction:
  - SimpleShear: corrects for simple shear experiment
  - UniAxial: corrects for uniaxial experiments
  - NoCorrection: the data are not corrected
- mpa: tells explicitly the unit of measure (to convert from MPa->Pa)
- d: is the reference grain size
- m: is the grain size exponent
- water_correction: Tells whether or not a water correction must be applied:
  - Fugacity : corrects the pre-exponential factor with water fugacity
  - COH: corrects the pre-exponential factor for Concentration.
- ref: hopefully the reference of the rheological flow law.


**dislocation**
```
  Dislocation_WetOlivine: 
    b: 1600
    e: 520.0e3
    v: 22e-6
    f: 'Simpleshear'
    mpa: 1
    r: 1.2
    n: 3.5
    b_si: 'MPa^-n s^-1 COH^-r'
    water_correction: 'COH'
    ref: 'Hirth, Greg, and David Kohlstedt. "Rheology of the upper mantle and the mantle wedge: A view from the experimentalists." Geophysical monograph series 138 (2003): 83-105.'
```
- n: stress exponent
- r: water exponent
  
The rheological database should be constructed introducing the original data, and flagging what the required corrections to apply are. For example: diffusion rheologies are the best fit of experimental data; this fit is made with a specific law that incorporate explicitly the grain size. **StonedFEniCSx** cannot handle grain size evolution, thus, the reference grain size is used to correct the pre-exponential factor and transforming into {math}`MPa^{-1}s^{-1}`, then, as a function of the type of experiment an additional correction is applied. If the experiments accounted water content, there is a small flag that tells whether or not the fitting has been carried out with fugacity laws or water concentration. The pre-exponential factor is then corrected with a reference water fugacity/concentration and then ultimately converted into the final unit of measure {math}`Pa^{-1}s^{-1}`.

Most of the time, typesetter or authors themselves are not caring so much about the unit of measure. If user wants to introduce his customize rheology, they needs to check the unit of measures. The code is designed to convert the unit of measure before the configuration stage, so, it is necessary to properly handle the unit of measure. 

In the following portion, the main rheologies in the code will be listed. Additionally, the rheology avaialable for the virtual shear zone will be described. 

**Common parameters:** n = 1.0, m = 0.0, d = 1.0, ah2o = 1.0, bh2o = 5521×10⁶, eh2o = 31.28×10³, vh2o = −2.009×10⁻⁵

### Diffusion Creep

| Name | b | e  [J/mol] | v  [m³/mol] | m | r | d [μm] | f (correction) | mpa | b_si | Water corr. | Ref (short) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `Hirth_dry_Dislocation_creep` | 1.5e9 | 375.0e3 | 5e-6 | 3.0 | 0 | 10e3 | Simpleshear | 1 | MPa⁻¹ s⁻¹ | None |{cite}`hirth2003rheology` |
| `Hirth_wet_Diffusion_creep` | 2.7e7 | 375.0e3 | 10e-6 | 3.0 | 0.8 | 10e3 | Simpleshear | 1 | MPa⁻¹ s⁻¹ COH⁻ʳ | COH |{cite}`hirth2003rheology` |
| `VK_Diffusion_creep` | 3.79e-10 | 335.0e3 | 0e-6 | 1.0 | 0.8 | 1.0 | None | 0 | Pa⁻¹ s⁻¹ | None |{cite}`van2008community` |

### Dislocation Creep

| Name | b | e  [J/mol] | v [m³/mol] | n | r | f (correction) | mpa | b_si | Water corr. | Ref (short) |
|---|---|---|---|---|---|---|---|---|---|---|
| `Hirth_dry_Dislocation_creep` | 1.1e5 | 345.0e3 | 15e-6 | 3.5 | 0.0 | Simpleshear | 1 | MPa⁻ⁿ s⁻¹ | None | {cite}`hirth2003rheology` |
| `Hirth_wet_Dislocation_creep` | 1600 | 520.0e3 | 22e-6 | 3.5 | 1.2 | Simpleshear | 1 | MPa⁻ⁿ s⁻¹ COH⁻ʳ | COH |{cite}`hirth2003rheology` |
| `VK_Dislocation_creep` | 2.136e-17 | 540.0e3 | 0.0 | 3.5 | 0.0 | None | 0 | MPa⁻ⁿ s⁻¹ COH⁻ʳ | None | {cite}`van2008community` |
| `Wet_Quartzite_2001_Dislocation_creep` | 2.7e7 | 345.0e3 | 38e-6 | 3.0 | 0.0 | Uniaxial | 1 | MPa⁻ⁿ s⁻¹ | None | {cite}`rybacki2004deformation` |
| `Hirareth_Serpentinite_Dislocation_creep` | 2.82e-15 | 8900 | 3.2e-6 | 3.8 | 0.0 | Uniaxial | 1 | MPa⁻ⁿ s⁻¹ | None | {cite}`hilairet2007high` |
| `Wet_Quartzite_2001_Dislocation_creep` | 6.31e-12 | 135.0e3 | 0e6 | 4.0 | 1.0 | Uniaxial | 1 | MPa⁻⁽ⁿ⁺ʳ⁾ s⁻¹ | Fugacity | {cite}`hirth2001evaluation` |
| `Glaucophane_2025_Dislocation_creep` | 2.32e10 | 450.0e3 | 0e-6 | 3.0 | 0.0 | Uniaxial | 1 | MPa⁻ⁿ s⁻¹ | None | {cite}`hufford2026blueschist`|


The viscosity is computed using the harmonic average: 

```{math}
    \eta_{eff} = (\eta_{dif}^{-1}+\eta_{dis}^{-1}+\eta_{max}^{-1})^{-1}
```
where {math}`\eta_{eff}` is the effective viscosity and {math}`eta_{max}` is the maximum viscosity (parameter that stabilises the numerical computation). There are two main scenarios: the only active mechanism is diffusion creep and the full composite one. In case of simulation with only diffusion creep, the harmonic average omits the dislocation creep viscosity. 

**Note**: The reference represents where I first encounter this rheology. For example, `VK_Diffusion_creep` is originally coming from {cite}`karato1993rheology`

### Conductivity

### Heat capacity

### Thermal expansivity

### Density

## References

```{bibliography}
:all:
```

