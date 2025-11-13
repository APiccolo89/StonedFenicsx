# StonedFenicsx

Kynematic thermal numerical code for describing slab temperature evolution

---

## Table of Contents
- [Introduction](#introduction)
- [NumericalMethods](#numerical methods)
- [Domain](#domain)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Introduction

---
## Numerical Methods

Stonedfenicsx is a numerical code that solves the continuity, momentum and energy conservation equation using FEM. The numerical model is fully driven by the kynematic boundary condition. The media 
is incompressible, and the momentum equation does not have any gravitational momentum source. The strong form of the mechanical equation is: 

- **Mass conservation**: $\nabla \cdot u = 0$ 
    - $u$: velocity field 
- **Momentum conservation**: $\nabla \cdot \mathbf \tau + \nabla \cdot P = 0$
    - $\tau$: *deviatoric stress*: $\tau = 2 \cdot \eta_{\mathrm{eff}} \cdot \dot{\mathbf{\varepsilon}}$
        - $\dot{\mathbf{\varepsilon}}$: *deviatoric strain tensor*: $\dot{\mathbf{\varepsilon}} = \frac{1}{2} \cdot (\nabla u + \nabla u ^{T}) $
        - $\eta_{\mathrm{eff}}$: *effective viscosity* : $\eta_{\mathrm{eff}} = f(T,P,\dot{\mathbf{\varepsilon}}_{II})$
            - $\dot{\varepsilon}_{II}$: *deviatoric strain rate second invariant*
The resolution of these equations gives a velocity and pressure field. The effective
viscosity ($\eta_{\mathrm{eff}}$) can depend on the temperature, pressure and strain rate second invariant. 

The code solves both the steady-state and time-dependent conservation of energy. 
- **Steady-state energy conservation equation**:   $\nabla \cdot (k \cdot \nabla \cdot T) + \rho \cdot C_p \cdot \nabla \cdot T + H= 0$
- **Time-dependent energy conservation equation**: $\rho \cdot C_p \cdot \frac{\partial T}{\partial t} + \rho \cdot C_p \cdot \nabla \cdot T + \nabla \cdot (k \cdot \nabla \cdot T)  + H = 0 $
  - $k$   : *Heat conductivity* [W m/ K ]: $k = f(P,T)$ 
  - $C_p$ : *Heat capacity*     [kJ/kg/K]: $C_p = f(T)$ 
  - $\rho$: *Density*            [$kg/m^3$]: $\rho = f(P,T)$       
  - $t$   : *time* 

The material properties required to solve both the steady-state and time-dependent equations can depends on temperature and pressure. Pressure depends on the velocity field, and without the gravitational momentum source is not reliable for computing the material properties. To compute the pressure and temperature dependent material properties, it is necessary to introduce a lithostatic pressure field. The lithostatic pressure field is computed following *Jourdon et al, 2022* [1]. The strong form for solving the lithostatic pressure field (i.e., steady state stokes) is: 

- **Lithostatic Pressure**: $\nabla \cdot \nabla \left P^{L}\right - \nabla \cdot \left(\rho g\right \right) = 0$
    - g: *gravity acceleration* 

This equation gives as results a lithostatic pressure field that can be used to compute the material property as a function of depth. 

---

## Installation

---

## Reference 
[1]: Jourdon, A. and May, D. A.,*An efficient partial-differential-equation-based method to compute pressure boundary conditions in regional geodynamic models* DOI: 10.5194/se-13-1107-2022
