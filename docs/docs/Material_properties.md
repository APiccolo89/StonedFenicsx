# Material properties

Material properties are defined using the input values specified in *input.yml* (see *Material properties* in *How to use*). **StonedFEniCSx** uses the options specified in *input.yml*, overwrites those that are not required (see below), and creates a small internal database. This database consists of arrays associated with specific properties, each with a size equal to the total number of phases. Internally, **StonedFEniCSx** accesses the properties of each subregion using its corresponding *ID*.

Inside the *config* folder (`stonedfenicsx/config`), there is a folder containing the material-property definitions and the corresponding dictionaries. These databases contain the original values of the material-property parameters. The parameters are converted into the appropriate units (e.g., MPa → Pa) and subsequently divided by the corresponding characteristic scales. This process is performed during the configuration stage of the numerical simulation.

(Rocks_ID)=

## Rock phases and IDs

The numerical domain is divided into three computational meshes: the overriding plate, subducting plate, and mantle wedge. These subdomains can contain one or more rock phases. The rock phases represent different lithologies, and their ID numbers link them to the material database. The mandatory phases are:

- `subducting plate mantle` (ID = 1): the subducting plate material.
- `wedge mantle` (ID = 3): the convective mantle.
- `overriding plate mantle` (ID = 4): the overriding mantle lithosphere.

The user can introduce crustal layers:

- `oceanic crust` (ID = 2): the oceanic crust of the subducting plate.
- `overriding upper crust` (ID = 5): an upper crustal layer of the overriding plate.
- `overriding lower crust` (ID = 6): a lower crustal layer of the overriding plate.

The customisation of the material properties of each phase depends on the subdomain to which the phase belongs. For example, the `oceanic crust` and `subducting plate mantle` always have a constant viscosity, but they can have different thermal properties (see **Tab.** {ref}`table:material_phases`). Therefore, within the subducting plate, the user's rheological choices are overwritten by the default viscosity. This design choice allows the code to be extended to other applications in the future.

(table:material_phases)=

| Phase name | Optional | Rheology {math}`\eta` | Thermal conductivity {math}`k` | Density {math}`\rho` | Heat capacity {math}`C_p` | Thermal expansion {math}`\alpha` | IDs |
|------------|----------|----------|----------------------|---------|---------------|---------------------|---|
| Subducting plate mantle | No | Constant viscosity | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear | 1 |
| Oceanic crust | Yes | Constant viscosity | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear | 2 |
| Wedge mantle | No | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear | 3 |
| Overriding plate mantle | No | Constant viscosity | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear | 4 |
| Overriding upper crust | Yes | Constant viscosity | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear | 5 |
| Overriding lower crust | Yes | Constant viscosity | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear | 6 |

## Material properties
In the following section, the non-linear properties will be described.
At the end of the pages, there is a succint code interface to call these non-linear properties. 

### Rheological material properties

Viscosity can be constant, temperature-dependent, or non-linear and temperature-dependent. The `wedge mantle` is the only phase for which different rheological models can be selected. Temperature-dependent viscosity is described by a diffusion-creep mechanism, while non-linear temperature-dependent viscosity is described by a dislocation-creep mechanism. The general equation for both mechanisms is:

```{math}
:label: eq:diffusion_dislocation_creep

\eta_{\mathrm{dif|dis}} =
B_{\mathrm{dif|dis}}
\, \dot{\varepsilon}_{II}^{\,1-\frac{1}{n}}
\exp\!\left(
-\frac{E_{\mathrm{dif|dis}} + P V_{\mathrm{dif|dis}}}{n R T}
\right)
```

{math}`B_{\mathrm{dif|dis}}` is the pre-exponential factor for either diffusion (`dif`) or dislocation (`dis`) creep. {math}`\dot{\varepsilon}_{II}` is the second invariant of the strain-rate tensor. {math}`n` is the stress exponent ({math}`n = 1` for diffusion creep). {math}`E_{\mathrm{dif|dis}}` and {math}`V_{\mathrm{dif|dis}}` are the activation energy and activation volume, respectively. {math}`T` and {math}`P` are temperature and pressure, respectively, and {math}`R` is the ideal gas constant.

The user can customise each parameter of the diffusion- and dislocation-creep laws. **StonedFEniCSx** contains an internal database that collects the available rheologies. A subset of the database is shown below to illustrate the structure of the diffusion- and dislocation-creep entries.

**Diffusion**

```yaml
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

This database is built using the original rheological data. It also contains a set of common parameters, such as the water-fugacity parameters.

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
- `ref`: reference for the rheological flow law

**Dislocation**

```yaml
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

The rheological database is constructed from the original experimental parameters, together with information specifying which corrections must be applied. For example, diffusion-creep rheologies are obtained by fitting experimental data to a flow law that explicitly incorporates grain size. **StonedFEniCSx** does not account for grain-size evolution; therefore, the reference grain size is used to correct the pre-exponential factor and convert it to {math}`MPa^{-1}s^{-1}`. An additional correction is then applied depending on the type of experiment. If the experiments account for water content, a flag specifies whether the fit was performed using water fugacity or water concentration. The pre-exponential factor is then corrected using a reference water fugacity or concentration and finally converted to {math}`Pa^{-1}s^{-1}`.

The units reported for rheological parameters are not always consistent across the literature. When introducing a custom rheology, the user should therefore verify the original units carefully. **StonedFEniCSx** performs the required unit conversions during the configuration stage, so the units of the original parameters must be specified correctly.

In general the correction factor for the pre-exponential factor of each of the rheological law has this form:
```{math}
:label: eq:correction_factor
b_{cor} = F W_0^r d_0^{-m}\Xi b 

```
where `F` is the correction for the type of experiment, {math}`W_0` is the reference quantity of water fugacity/concentration, {math}`d_0` is the reference grain size, {math}`\Xi` is the conversion factor from MPa to Pa, and `b` is the original pre-exponential factor. 


The main rheologies available in the code are listed in the following section. The rheology available for the virtual shear zone is also described.

#### Tables

**Common parameters:** n = 1.0, m = 0.0, d = 1.0, ah2o = 1.0, bh2o = 5521×10⁶, eh2o = 31.28×10³, vh2o = −2.009×10⁻⁵

`Name` is the actual string that must be used in the *input.yml* file.

::::{tab-set}

:::{tab-item} Diffusion Creep

```{table} Diffusion creep
:name: Table_Diffusion_Creep

| Name | b | e [J/mol] | v [m³/mol] | m | r | d [μm] | f (correction) | mpa | b_si | Water corr. | Ref (short) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `Hirth_dry_Diffusion_creep` | 1.5e9 | 375.0e3 | 5e-6 | 3.0 | 0 | 10e3 | Simpleshear | 1 | MPa⁻¹ s⁻¹ | None | {cite}`hirth2003rheology` |
| `Hirth_wet_Diffusion_creep` | 2.7e7 | 375.0e3 | 10e-6 | 3.0 | 0.8 | 10e3 | Simpleshear | 1 | MPa⁻¹ s⁻¹ COH⁻ʳ | COH | {cite}`hirth2003rheology` |
| `VK_Diffusion_creep` | 3.79e-10 | 335.0e3 | 0e-6 | 1.0 | 0.8 | 1.0 | None | 0 | Pa⁻¹ s⁻¹ | None | {cite}`van2008community` |
```
:::

:::{tab-item} Dislocation Creep
```{table} Dislocation creep
:name: Table_Dislocation_Creep

| Name | b | e [J/mol] | v [m³/mol] | n | r | f (correction) | mpa | b_si | Water corr. | Ref (short) |
|---|---|---|---|---|---|---|---|---|---|---|
| `Hirth_dry_Dislocation_creep` | 1.1e5 | 345.0e3 | 15e-6 | 3.5 | 0.0 | Simpleshear | 1 | MPa⁻ⁿ s⁻¹ | None | {cite}`hirth2003rheology` |
| `Hirth_wet_Dislocation_creep` | 1600 | 520.0e3 | 22e-6 | 3.5 | 1.2 | Simpleshear | 1 | MPa⁻ⁿ s⁻¹ COH⁻ʳ | COH | {cite}`hirth2003rheology` |
| `VK_Dislocation_creep` | 2.136e-17 | 540.0e3 | 0.0 | 3.5 | 0.0 | None | 0 | MPa⁻ⁿ s⁻¹ COH⁻ʳ | None | {cite}`van2008community` |
| `Wet_Quartzite_2001_Dislocation_creep` | 2.7e7 | 345.0e3 | 38e-6 | 3.0 | 0.0 | Uniaxial | 1 | MPa⁻ⁿ s⁻¹ | None | {cite}`rybacki2004deformation` |
| `Hilairet_Serpentinite_Dislocation_creep` | 2.82e-15 | 8900 | 3.2e-6 | 3.8 | 0.0 | Uniaxial | 1 | MPa⁻ⁿ s⁻¹ | None | {cite}`hilairet2007high` |
| `Wet_Quartzite_2001_Dislocation_creep` | 6.31e-12 | 135.0e3 | 0e-6 | 4.0 | 1.0 | Uniaxial | 1 | MPa⁻⁽ⁿ⁺ʳ⁾ s⁻¹ | Fugacity | {cite}`hirth2001evaluation` |
| `Glaucophane_2025_Dislocation_creep` | 2.32e10 | 450.0e3 | 0e-6 | 3.0 | 0.0 | Uniaxial | 1 | MPa⁻ⁿ s⁻¹ | None | {cite}`hufford2026blueschist` |
```
:::

::::

The viscosity is computed using the harmonic average:

```{math}
\eta_{\mathrm{eff}} =
\left(
\eta_{\mathrm{dif}}^{-1}
+
\eta_{\mathrm{dis}}^{-1}
+
\eta_{\mathrm{max}}^{-1}
\right)^{-1}
```

where {math}`\eta_{\mathrm{eff}}` is the effective viscosity and {math}`\eta_{\mathrm{max}}` is the maximum viscosity, a parameter used to stabilise the numerical computation. Two rheological configurations are available: diffusion creep alone or the full composite rheology. When only diffusion creep is active, the dislocation-creep contribution is omitted from the harmonic average.

#### Water fugacity and concentration

The parameters of some rheological laws have been fitted using experimental data that accounted water concentration/fugacity. Water quantity is not accounted in **StonedFEniCSx**. The easiest solution for correcting the pre-exponential factor (which can be interpreted as a trashbin of all the unit of measure of an experimental fitting) is to compute a reference water quantity and multiply the original value for it. However, for certain dislocation flow law was necessary to introduce an additional parameters (i.e., `Wet Quartzite`) to guarantee the reproducibility of the experiments. `Wet Quartzite` is used mainly to describe the rheology of the subduction interface; and it accounts in a questionable way the effects of the fugacity of water as a function of pressure-temperature. To mantain a rigorous treatment of the material property, a P-T correction factor is applied to the viscosity of the shear zone:

```{math}
:label: eq:water_correction

\begin{aligned}
W_0 &=
a_{H_2O} B_{fH_2O}
\exp\left(
-\frac{E_{fH_2O} + P_{\mathrm{ref}}V_{fH_2O}}
{RT_{\mathrm{ref}}}
\right), \\[6pt]
W_i &=
a_{H_2O} B_{fH_2O}
\exp\left(
-\frac{E_{fH_2O} + PV_{fH_2O}}
{RT}
\right), \\[6pt]
\frac{W_i}{W_0} = \zeta &=
\exp\left(
\frac{
-RT_{\mathrm{ref}}(E_{fH_2O}+PV_{fH_2O})
+RT(E_{fH_2O}+P_{\mathrm{ref}}V_{fH_2O})
}
{RTT_{\mathrm{ref}}}
\right), \\[6pt]
W_i &= \zeta W_0.
\end{aligned}
```

The reference state is $T_{\mathrm{ref}} = 298.15$ K and
$P_{\mathrm{ref}} = 10^5$ Pa. Since $W_0$ is already accounted for in the
corrected pre-exponential factor, it is only necessary to multiply the
rheological equations by $\zeta$, which describes the variation with respect
to the reference state.

>[!NOTE]
>The reference indicates the source from which a particular rheology was first introduced into **StonedFEniCSx**, rather than necessarily the original publication of the flow law. For >example, `VK_Diffusion_creep` ultimately originates from {cite}`karato1993rheology`.


### Thermal properties

The implementation of the thermal properties follows {cite}`richards2020structure`, {cite}`grose2013comprehensive`, and {cite}`korenaga2016evolution`. The material properties are computed using the formulations presented in these studies, and the corresponding parameter values are taken from these publications.

In general, the code can handle pressure-dependent material properties. However, these should be used with caution because the kinematic subduction models are incompressible. Consequently, pressure-dependent properties may introduce inconsistencies in the computed thermal field. Adiabatic heating cannot be introduced straightforwardly within this framework. Furthermore, the pressure singularities described by {cite}`van2008community` complicate the computation of shear heating without introducing additional assumptions or arbitrary choices.

#### Heat capacity
Heat capacity is computed using the following equation:

```{math}
:label: eq:heat_capacity

\begin{aligned}
C_p(X,T) ={}& cp0(X)
+ cp1(X)T^{-0.5}
+ cp2(X)T^{-2} \\
&+ cp3(X)T^{-3}
+ cp4(X)T
+ cp5(X)T^{2}
\end{aligned}
```

where {math}`cp{i}` are empirical parameters that depend on the mineralogical composition. The data are taken from {cite}`grose2013comprehensive` and {cite}`berman1996optimized`. The main parameters used in the current implementation are listed in {ref}`tab:heat_capacity`.

The coefficients are defined such that $C_p$ is expressed in {math}`\mathrm{J\,kg^{-1}\,K^{-1}}`, with T in K. Forsterite and fayalite are the end-member experimental data from {cite}`berman1996optimized` or {cite}`berman1988internally`, while olivine, augite, and plagioclase are the mineral properties listed in {cite}`grose2013comprehensive`. In Tab {numref}`tab-original_data`, the original experimental data are listed, while In Tab {numref}`tab-mix_data` the mixture that can be used are listed (using the abbreviation in the name of {numref}`tab-original_data`)

>[!NOTE]
> The olivine listed in {cite}`grose2013comprehensive` it is a mixture computed using the data from {cite}`berman1996optimized`. While the data of augite and plagioclase, instead, are tabulated from {cite}`robie1995thermodynamic` (NB: in {cite}`grose2013comprehensive` this is reference is not listed, so the reference here can be wrong.)
> The equation of the heat capacity is not canonical, it is simply the most generalised form that have been conceived to collects all the parameters found in literature. Thus, the reader must map each of the coefficient as a function of the temperature exponent. This was a design solution to avoid to have cases as a function of the mineralogical composition. 


The heat capacity of the mantle is computed using a mixture of 0.9 forsterite and 0.1 fayalite. The crustal heat capacity is computed using a mixture of 0.65 plagioclase, 0.2 augite, and 0.15 olivine, as listed in {cite}`grose2013comprehensive`. 


::::{tab-set}

:::{tab-item} Original data 

```{table} Heat capacity
:name: tab-original_data
| Model                             | cp0      | cp1          | cp2       | cp3             | cp4        | cp5       | ref   |
|-----------------------------------|----------|--------------|-----------|-----------------|------------|-----------|-------
| Bermann_Fosterite (**BFo**)           | 1696.199 | -14224.790   | 0.0       | -8.262078e8     | 0.0        | 0.0       |{cite}`berman1988internally`|
| Bermann_Fayalite (**BFa**)           | 1221.616 | -9441.481    | 0.0       | -6.826290e8     | 0.0        | 0.0       |{cite}`berman1988internally`|
| Bermann_Aranovich_Fosterite (**BAFo**)| 1657.391 | -12805.368   | 0.0       | -1.904457e9     | 0.0        | 0.0       |{cite}`berman1996optimized`|
| Bermann_Aranovich_Fayalite (**BAFa**)| 1236.682 | -9882.172    | 0.0       | -3.051955e8     | 0.0        | 0.0       |{cite}`berman1996optimized`|
| olivine (**ol**)                      | 1610.8   | -12478.8     | 0.0       | -1.728477e9     | 0.0        | 0.0       |{cite}`grose2013comprehensive`|
| augite (**au**)                       | 2171.5   | -22271.6     | 1.1333e6  | 0.0             | -4.555e-1  | 1.299e-4  |{cite}`grose2013comprehensive`|
| plagioclase (**plg**)                 | 1857.57  | -16494.6     | -5.061e6  | 0.0             | -3.324e-1  | 1.505e-4  |{cite}`grose2013comprehensive`|
```
:::

:::{tab-item} Avaiable Mixture
```{table} Avaiable mixture
:name: tab-mix_data
| Name input | Mixture |
|---|---|
| `Mantle_Bernard_1988_FO` | *BFo* 100% |
| `Mantle_Bernard_1988_FA` | *BFa* 100% |
| `Mantle_Bernard_Ar_199x_FO` | *BAFo* 100% |
| `Mantle_Bernard_Ar_199x_FA` | *BAFa* 100% |
| `Mantle_Bernard_1988_FO_FA` | *BFo* 90% + *BFa*10%  |
| `Mantle_Bernard_Ar_199x_FO_FA` |  *BAFo* 90% + *BAFa*10%  |
| `Oceanic_crust` | *ol*15% + *au*20% +*pl*65%|
```

:::

::::

#### Thermal expansivity

#### Density

#### Conductivity

Thermal conductivity has two contributions: lattice conductivity, obtained from the lattice diffusivity, and radiative conductivity.

##### Lattice Diffusivity

The lattice diffusivity is computed according to the following equation:

```{math}
:label: lattice_diffusivity

\kappa_{\mathrm{lat}} =
\kappa_0
+
\kappa_1 \exp(\frac{-(T-T_{\mathrm{ref}})}{T_0})
+
\kappa_2 \exp(\frac{-(T-T_{\mathrm{ref}})}{T_1})
\exp(fP)
```

The parameters used in this formulation are compiled from {cite}`richards2020structure`, {cite}`grose2013comprehensive`, and {cite}`korenaga2016evolution` and are listed in {ref}`Table_lattice_dif`. The crustal lattice diffusivity is computed by combining the mantle, augite, and anorthite (`AnAb`) parameterisations. The oceanic crust is assumed to consist of 65% plagioclase, 15% olivine, and 20% clinopyroxene, consistent with the calculation of heat capacity.

The lattice diffusivity is then multiplied by the density and heat capacity to obtain the corresponding lattice thermal conductivity.

##### Radiative Conductivity

{math}`k_{\mathrm{rad}}` is described by the following equation {cite:p}`grose2013comprehensive,richards2020structure`:

```{math}
:label: eq:radiative_conductivity

k_{\mathrm{rad}} =
A_r \exp\left(-\frac{(T-T_a)^2}{2x_a^2}\right)
+
B_r \exp\left(-\frac{(T-T_b)^2}{2x_b^2}\right)
```

where {math}`A_r`[W/m/K], {math}`B_r` [W/m/K], {math}`T_a` [K], {math}`x_a` [K], {math}`x_b` [K], and {math}`T_b` [K] are computed using the grain size *d*:

```{math}
:label: eq:radiative_parameters

\begin{aligned}
A_r &= 1.8 \left[1-\exp\left(-\frac{d^{1.3}}{0.15}\right)\right]
      - \left[1-\exp\left(-\frac{d^{0.5}}{5}\right)\right], \\[6pt]
B_r &= 11.7 \exp\left(-\frac{d}{0.159}\right)
      + 6 \exp\left(-\frac{d^{3}}{10}\right), \\[6pt]
T_a &= 490
      + 1850 \exp\left(-\frac{d^{0.315}}{0.825}\right)
      + 875 \exp\left(-\frac{d}{0.18}\right), \\[6pt]
T_b &= 2700
      + 9000 \exp\left(-\frac{d^{0.5}}{0.205}\right), \\[6pt]
x_a &= 167.5
      + 505 \exp\left(-\frac{d^{0.5}}{0.85}\right), \\[6pt]
x_b &= 465
      + 1700 \exp\left(-\frac{d^{0.94}}{0.175}\right).
\end{aligned}
```

The default grain size {math}`d` is 0.5 cm. This set of equations is given by {cite:p}`grose2013comprehensive` and is based on the work of {cite:p}`hofmeister2005dependence`. In {cite:p}`hofmeister2005dependence`, radiative conductivity is described by three equations applicable to different grain-size ranges. The formulation of {cite:p}`grose2013comprehensive` provides a convenient unified parameterisation and has subsequently been used to model the thermal evolution of cooling oceanic lithosphere in {cite:p}`richards2020structure` and {cite:p}`korenaga2016evolution`.

##### Effective thermal conductivity

The final thermal conductivity is computed as:

```{math}
:label: final_conductivity

k(T,P,X)
=
\kappa_{\mathrm{lat}}(T,P,X)
\rho(P,T,X)
C_p(T,X)
+
k_{\mathrm{rad}}(T)
```
(Table_lattice_dif)=

 *Experimental data of thermal diffusivity*

| Name | {math}`\kappa_0` [mm<sup>2</sup>/s] | {math}`\kappa_1` [mm<sup>2</sup>/s] | {math}`T_1` [K] | {math}`\kappa_2` [mm<sup>2</sup>/s] | {math}`T_2`[K] | f [Pa<sup>-1</sup>] |
|---|---:|---:|---:|---:|---:|---:|
| `Mantle_Richards_2018` | 0.565e-6 | 0.67e-6 | 590.0 | 1.4e-6 | 135.0 | 0.05e-9 |
| `Augite` | 0.59e-6 | 1.03e-6 | 386.0 | 0.928e-6 | 125.0 | 0.05e-9 |
| `AnAb` | 0.36e-6 | 0.4e-6 | 300.0 | 0.0 | 1.0 | 0.05e-9 |
| `Crust_Richards_2018` | 0.432e-6 | 0.44e-6 | 380 | 0.305e-6 | 145.0 | 0.05e-9 |

