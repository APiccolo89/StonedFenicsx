# Material properties

Material properties are defined using the input values defined in the *input.yml* (*Material properties* in *How to use*). **StonedFEniCSx** uses the option listed defined in the *input.yml*, over-writes the options that are not needed (see below), and create a small database. The small database is a small collection of arrays associated to specific properties, featuring a size equal to the total number of phases. Internally, **StonedFEniCSx** access to the specific property using the *ID* number of the subregions. 

Inside *config* folder  (`\stonedfenicsx\config`) there is a folder containing the material properties and the relative dictionaries. These databases contain the original value of the material properties parameters; these parameters are always converted into the suitable unit of measure (e.g., MPa -> Pa) and then they are divided by the characteristic scales. This process is always done during the configuration stage of the numerical simulation. 

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

User can customise each of the parameter of diffusion and dislocation creep. In StonedFEniCSx there are few internal rheological database, that can be instantiated in the input file: 
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


