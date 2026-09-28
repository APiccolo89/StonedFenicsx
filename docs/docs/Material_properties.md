# Material properties

Material properties are defined using the input values specified in *input.yml* (*Material properties* in *How to use*). **StonedFEniCSx** uses the options listed in *input.yml*, overwrites the options that are not needed (see below), and creates a small database. This database is a collection of arrays associated with specific properties, with a size equal to the total number of phases. Internally, **StonedFEniCSx** accesses a specific property using the *ID* number of the corresponding subregion.

Inside the *config* folder (`stonedfenicsx/config`), there is a folder containing the material properties and the corresponding dictionaries. These databases contain the original values of the material-property parameters. These parameters are always converted into the appropriate units (e.g., MPa → Pa) and then divided by the characteristic scales. This process is always performed during the configuration stage of the numerical simulation.

(Rocks_ID) =

## Rock phases and IDs

The numerical domain is divided into three different computational meshes: the overriding plate, subducting plate, and mantle wedge. These subdomains can be composed of one or more rock phases. The rock phases represent different lithologies, and their ID numbers connect them to the material database. The mandatory phases are:

- `subducting plate mantle` (ID = 1): the subducting plate material.
- `wedge mantle` (ID = 3): the convective mantle.
- `overriding plate mantle` (ID = 4): the overriding mantle lithosphere.

The user can introduce crustal layers:

- `oceanic crust` (ID = 2): the oceanic crust of the subducting plate.
- `overriding upper crust` (ID = 5): an upper crustal layer for the overriding plate.
- `overriding lower crust` (ID = 6): a lower crustal layer for the overriding plate.

The customisation of the material properties of each phase depends on the subdomain to which the phase belongs. For example, `oceanic crust` and `subducting plate mantle` always have a constant viscosity, but they can have different thermal material properties (see **Tab.** {ref}`table:material_phases`). This means that, within the subducting plate, the user's rheological choices are overwritten by the default viscosity. This design choice allows the code to be extended to additional purposes in the future.

(table:material_phases)=

| Phase name | Optional | Rheology {math}`\eta`| Thermal conductivity {math}`k`| Density {math}`\rho`| Heat capacity {math}`C_p`| Thermal expansion {math}`\alpha` | IDs |
|------------|----------|----------|----------------------|---------|---------------|---------------------|---|
| Subducting plate mantle | No | Constant viscosity | Linear / non-linear| Linear / non-linear | Linear / non-linear | Linear / non-linear | 1|
| Oceanic crust | Yes | Constant viscosity | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear |2|
| Wedge mantle | No | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear |3|
| Overriding plate mantle | No | Constant viscosity | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear | 4|
| Overriding upper crust | Yes | Constant viscosity | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear |5|
| Overriding lower crust | Yes | Constant viscosity | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear|6|

## Material properties

### Rheological material properties

Viscosity can be either constant, temperature-dependent, or non-linear and temperature-dependent. The only phase that can use different rheological models is the `wedge mantle`. Temperature-dependent viscosity is described by a diffusion-creep mechanism, while non-linear temperature-dependent viscosity is described by a dislocation-creep mechanism. The general equation for both mechanisms is:

```{math}
:label: eq:diffusion_dislocation_creep

\eta_{\mathrm{dif|dis}} =
B_{\mathrm{dif|dis}}
\, \dot{\varepsilon}_{II}^{\,1-\frac{1}{n}}
\exp\!\left(
-\frac{E_{\mathrm{dif|dis}} + P V_{\mathrm{dif|dis}}}{n R T}
\right)
```

{math}`B_{dif|dis}` is the pre-exponential factor for either diffusion (`dif`) or dislocation (`dis`) creep. {math}`\dot{\varepsilon}_{II}` is the second invariant of the strain-rate tensor. {math}`n` is the stress exponent ({math}`n = 1` in the case of diffusion creep). {math}`E_{\mathrm{dif|dis}}` and {math}`V_{\mathrm{dif|dis}}` are the activation energy and activation volume, respectively. {math}`T` and {math}`P` are temperature and pressure, respectively, and {math}`R` is the ideal gas constant.

The user can customise each parameter of the diffusion- and dislocation-creep laws. **StonedFEniCSx** contains an internal database that collects the available rheologies. A small portion of the database is shown below to illustrate the diffusion- and dislocation-creep entries:

**diffusion**

```
Common:
  n: 1.0
  m: 0.0
  d: 1.0
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

This database is built using the original rheological data. There are also a few common parameters (e.g., the water-fugacity parameters).

- `b`: pre-exponential factor
- `e`: activation energy
- `v`: activation volume
- `f`: correction:
  - `SimpleShear`: corrects for simple-shear experiments
  - `UniAxial`: corrects for uniaxial experiments
  - `NoCorrection`: the data are not corrected
- `mpa`: explicitly specifies the unit system and whether conversion from MPa to Pa is required
- `d`: reference grain size
- `m`: grain-size exponent
- `water_correction`: specifies whether a water correction must be applied:
  - `Fugacity`: corrects the pre-exponential factor using water fugacity
  - `COH`: corrects the pre-exponential factor using water concentration
- `ref`: reference for the rheological flow law.

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

- `n`: stress exponent
- `r`: water exponent

The rheological database should be constructed by introducing the original data and specifying which corrections need to be applied. For example, diffusion-creep rheologies are obtained by fitting experimental data. This fit uses a specific law that explicitly incorporates grain size. **StonedFEniCSx** cannot handle grain-size evolution; therefore, the reference grain size is used to correct the pre-exponential factor and transform it into {math}`MPa^{-1}s^{-1}`. Then, depending on the type of experiment, an additional correction is applied. If the experiments account for water content, a flag specifies whether the fitting was carried out using water-fugacity laws or water concentration. The pre-exponential factor is then corrected using a reference water fugacity/concentration and ultimately converted into the final unit {math}`Pa^{-1}s^{-1}`.

In many cases, typesetters or authors themselves do not pay much attention to the units of measure. If the user wants to introduce a custom rheology, they need to check the units carefully. The code is designed to convert the units before the configuration stage; therefore, it is necessary to specify the original units correctly.

In the following section, the main rheologies available in the code are listed. Additionally, the rheology available for the virtual shear zone is described.

#### Tables

**Common parameters:** n = 1.0, m = 0.0, d = 1.0, ah2o = 1.0, bh2o = 5521×10⁶, eh2o = 31.28×10³, vh2o = −2.009×10⁻⁵

`Name` is the actual string that must be used in the *input.yml* file.

##### Diffusion Creep

| Name | b | e [J/mol] | v [m³/mol] | m | r | d [μm] | f (correction) | mpa | b_si | Water corr. | Ref (short) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `Hirth_dry_Dislocation_creep` | 1.5e9 | 375.0e3 | 5e-6 | 3.0 | 0 | 10e3 | Simpleshear | 1 | MPa⁻¹ s⁻¹ | None | {cite}`hirth2003rheology` |
| `Hirth_wet_Diffusion_creep` | 2.7e7 | 375.0e3 | 10e-6 | 3.0 | 0.8 | 10e3 | Simpleshear | 1 | MPa⁻¹ s⁻¹ COH⁻ʳ | COH | {cite}`hirth2003rheology` |
| `VK_Diffusion_creep` | 3.79e-10 | 335.0e3 | 0e-6 | 1.0 | 0.8 | 1.0 | None | 0 | Pa⁻¹ s⁻¹ | None | {cite}`van2008community` |

##### Dislocation Creep

| Name | b | e [J/mol] | v [m³/mol] | n | r | f (correction) | mpa | b_si | Water corr. | Ref (short) |
|---|---|---|---|---|---|---|---|---|---|---|
| `Hirth_dry_Dislocation_creep` | 1.1e5 | 345.0e3 | 15e-6 | 3.5 | 0.0 | Simpleshear | 1 | MPa⁻ⁿ s⁻¹ | None | {cite}`hirth2003rheology` |
| `Hirth_wet_Dislocation_creep` | 1600 | 520.0e3 | 22e-6 | 3.5 | 1.2 | Simpleshear | 1 | MPa⁻ⁿ s⁻¹ COH⁻ʳ | COH | {cite}`hirth2003rheology` |
| `VK_Dislocation_creep` | 2.136e-17 | 540.0e3 | 0.0 | 3.5 | 0.0 | None | 0 | MPa⁻ⁿ s⁻¹ COH⁻ʳ | None | {cite}`van2008community` |
| `Wet_Quartzite_2001_Dislocation_creep` | 2.7e7 | 345.0e3 | 38e-6 | 3.0 | 0.0 | Uniaxial | 1 | MPa⁻ⁿ s⁻¹ | None | {cite}`rybacki2004deformation` |
| `Hirareth_Serpentinite_Dislocation_creep` | 2.82e-15 | 8900 | 3.2e-6 | 3.8 | 0.0 | Uniaxial | 1 | MPa⁻ⁿ s⁻¹ | None | {cite}`hilairet2007high` |
| `Wet_Quartzite_2001_Dislocation_creep` | 6.31e-12 | 135.0e3 | 0e6 | 4.0 | 1.0 | Uniaxial | 1 | MPa⁻⁽ⁿ⁺ʳ⁾ s⁻¹ | Fugacity | {cite}`hirth2001evaluation` |
| `Glaucophane_2025_Dislocation_creep` | 2.32e10 | 450.0e3 | 0e-6 | 3.0 | 0.0 | Uniaxial | 1 | MPa⁻ⁿ s⁻¹ | None | {cite}`hufford2026blueschist` |

The viscosity is computed using the harmonic average:

```{math}
\eta_{eff} = (\eta_{dif}^{-1}+\eta_{dis}^{-1}+\eta_{max}^{-1})^{-1}
```

where {math}`\eta_{eff}` is the effective viscosity and {math}`\eta_{max}` is the maximum viscosity, a parameter used to stabilise the numerical computation. There are two main scenarios: diffusion creep is the only active mechanism, or the full composite rheology is used. For simulations with diffusion creep only, the harmonic average omits the dislocation-creep viscosity.

**Note:** The reference indicates where I first encountered a particular rheology. For example, `VK_Diffusion_creep` originally comes from {cite}`karato1993rheology`.

### Thermal properties

The code follows the descriptions of thermal properties given by {cite}`richards2020structure`, {cite}`grose2013comprehensive`, and {cite}`korenaga2016evolution`. The material-property calculations follow the equations presented in these sources, and the parameter values have been taken from these publications.

In general, the code can handle pressure-dependent properties. However, the user should be careful when using them because the kinematic subduction models are incompressible. Therefore, there is a concrete risk of over- or underestimating the thermal field. Adiabatic heating cannot easily be introduced within this framework. The pressure singularities described by {cite}`van2008community` prevent the computation of shear heating without introducing several inconsistencies or arbitrary choices.

#### Heat capacity

#### Thermal expansivity

#### Density

#### Conductivity

## References

```{bibliography}
:all:
```