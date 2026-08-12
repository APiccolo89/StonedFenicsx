# Material properties

## Rock phase and IDs

The computational domain is divided into subdomains and subregions. Subregions are identified by an ID number that represents the rock phase and is assigned to each element belonging to the subregion. Subdomains are defined using the geometrical input information. They can contain different subregions with different IDs. However, each subdomain has its own set of IDs. IDs are static and they are not advected. Each ID number is used to connect the element to a specific material database. This strategy allows computing on the fly the material properties during the solution of the FEM problems.

The global domain can be divided into six subregions. The IDs of these six subregions are determined by the rock phase that they represent. Three of these six subregions are mandatory:

- `subducting plate mantle` (ID = 1): the subducting plate materials.
- `wedge mantle` (ID = 3): convective mantle.
- `overriding plate mantle` (ID = 4): the overriding mantle lithosphere.

The user can introduce crustal levels:

- `oceanic crust` (ID = 2): oceanic crust of the subducting plate.
- `overriding upper crust` (ID = 5): an upper crust layer for the overriding plate.
- `overriding lower crust` (ID = 6): a lower crust layer for the overriding plate.

The numbering of the IDs is ordered following the generation of the computational domain workflow. `subducting plate mantle` and `oceanic crust` (optional) compose the `Subducting plate domain`. `overriding plate mantle`, `overriding upper crust` (optional), and `overriding lower crust` (optional) compose the `Overriding plate domain`. `wedge mantle` composes the `Wedge domain`. The customisation of the material properties of each phase depends on the subdomain to which these phases belong. For example, `oceanic crust` and `subducting plate mantle` have a constant viscosity but can have different thermal material properties (see **Tab.** {ref}`table:material_phases`).

Users can define an additional phase: `weak zone`. This phase is not associated with any subregion and is used only to describe the weak interface between the `Overriding plate domain` and the `Subducting plate domain`.


(table:material_phases)=
| Phase name | Optional | Rheology {math}`\eta`| Thermal conductivity {math}`k`| Density {math}`\rho`| Heat capacity {math}`C_p`| Thermal expansion {math}`\alpha` | IDs |
|------------|----------|----------|----------------------|---------|---------------|---------------------|---|
| Subducting plate mantle | No | Constant viscosity | Linear / non-linear| Linear / non-linear | Linear / non-linear | Linear / non-linear | 1|
| Oceanic crust | Yes | Constant viscosity | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear |2|
| Overriding plate mantle | No | Constant viscosity | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear | 3|
| Wedge mantle | No | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear |4|
| Overriding upper crust | Yes | Constant viscosity | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear |5|
| Overriding lower crust | Yes | Constant viscosity | Linear / non-linear | Linear / non-linear | Linear / non-linear | Linear / non-linear|6|
| Virtual subduction channel | Yes | Linear (shear heating only) | Linear / non-linear | Linear / non-linear| Linear / non-linear | Linear / non-linear |7|

## Material properties

### Rheological material properties

Viscosity can be either constant, temperature dependent or non-linear temperature dependent. The only two phases that can access to different rheological models are:`wedge mantle` and `virtual subduction channel`. The temperature dependent viscosity is described by the diffusion creep mechanism, while the non-linear temperature dependent viscosity is described by the dislocation creep mechanism. The general equation for both of the mechanisms is:   
```{math}
:label: eq:diffusion_dislocation_creep
\eta_{\mathrm{dif|dis}} =
B_{\mathrm{dif|dis}}
\, \dot{\varepsilon}_{II}^{\,1-\frac{1}{n}}
\exp\!\left(
-\frac{E_{\mathrm{dif|dis}} + P V_{\mathrm{dif|dis}}}{n R T}
\right)
```
where {math}`B_{dif|dis}` is the pre-exponential factor for either diffusion (dif) or dislocation creep. {math}`\dot{\varepsilon}_{II}` is the second invariant of the strain rate tensor. {math}`n` is the stress-exponent ({math}`n = 1` in case of diffusion creep mechanism). {math}`E_{\mathrm{dif|dis}}` and {math}`V_{\mathrm{dif|dis}}`. {math}`T` and {math}`P` are the temperature and pressure and {math}`R` is the perfect gas constant. 

User can costumise each of the parameter of diffusion and dislocation creep. In StonedFEniCSx there are few internal rheological database, that can be instantiated in the input file: 
(table:rheological_flow_law)=
(table:rheological_flow_law)=
| Rheological flow law | {math}`B_{dif}`<br>{math}`\scriptstyle Pa^{-1}\,s^{-1}` | {math}`E_{dif}`<br>{math}`\scriptstyle J\,mol^{-1}` | {math}`\scriptstyle V_{dif}`<br>{math}`\scriptstyle(m^{3}\,mol^{-1})` | {math}`n`<br>n.d. | {math}`B_{dis}`<br>{math}`\scriptstyle Pa^{-n}\,s^{-1}` | {math}`E_{dis}`<br>{math}`\scriptstyle J\,mol^{-1}` | {math}`V_{dis}`<br>{math}`\scriptstyle m^{3}\,mol^{-1}` |
|----------------------|-----------------------------------------------|-------------------------------------------|---------------------------------------------|-----------|-----------------------------------------------|-------------------------------------------|---------------------------------------------|
| Hirth_Dry_Olivine_diff |  | 375e3 | 5e-6 | / | / | / | / |
| Hirth_Dry_Olivine_disl |  |  |  |  |  |  |  |
| Van_Keken_diff         |  |  |  |  |  |  |  |
| Van_Keken_disl         |  |  |  |  |  |  |  |
| Hirth_Wet_Olivine_diff |  |  |  |  |  |  |  |
| Hirth_Wet_Olivine_disl |  |  |  |  |  |  |  | 
| Wet_Plagioclase_diff   |  |  |  |  |  |  |  |
| Wet_Plagioclase_disl   |  |  |  |  |  |  |  |

The viscosity is computed using the harmonic average: 

```{math}
    \eta_{eff} = (\eta_{dif}^{-1}+\eta_{dis}^{-1}+\eta_{max}^{-1})^{-1}
```
where {math}`\eta_{eff}` is the effective viscosity and {math}`eta_{max}` is the maximum viscosity (parameter that stabilises the numerical computation). There are two main scenarios: the only active mechanism is diffusion creep and the full composite one. In case of simulation with only diffusion creep, the harmonic average omits the dislocation creep viscosity. 

### Conductivity

### Heat capacity

### Thermal expansivity

### Density

% Reference


