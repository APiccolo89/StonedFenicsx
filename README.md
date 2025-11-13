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

---

## Introduction

---
## Numerical Methods

Stonedfenicsx is a numerical code that solves the continuity, momentum and energy conservation equation using FEM. The numerical model is fully driven by the kynematic boundary condition. The media 
is incompressible, and the momentum equation does not have any gravitational momentum source. The strong form of the mechanical equation is: 

- \textbf{Mass conservation}: $\nabla \cdot u = 0$ 
    - $u$: velocity field 
- \textbf{Momentum conservation}: $\nabla \cdot \mathbf \tau + \nabla \cdot P = 0$
    - $\tau$ deviatoric stress: $\tau = 2 \cdot \eta_{\mathrm{eff}} \cdot \dot{\mathbf{\varepsilon}}$
        - $\dot{\mathbf{\varepsilon}}$ deviatoric strain tensor: $\dot{\mathbf{\varepsilon}} = \frac{1}{2} \cdot (\nabla u + \nabla u ^{T}) $
        - $\eta_{\mathrm{eff}}$ effective viscosity: $\eta_{\mathrm{eff}} = f(T,P,\dot{\mathbf{\varepsilon}}_{II})$
            -  $\dot{\mathbf{\varepsilon}}_{II}$: deviatoric strain rate second invariant $\dot{\mathbf{\varepsilon}}_{II} = \sqrt{ \frac{1}{2} \cdot \dot{\mathbf{\varepsilon}} : \dot{\mathbf{\varepsilon}}}$ 

The resolution of these equations gives a velocity and pressure field. Pressure depends on the velocity field, and without the gravitational momentum source is not reliable for computing the material properties. The effective
viscosity ($\eta_{\mathrm{eff}}$) can depend on the temperature, pressure and strain rate second invariant. 


---

## Installation

Clone the repository:

```bash
git clone https://github.com/username/project-name.git
cd project-name