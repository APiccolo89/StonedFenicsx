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

Stonedfenicsx is a numerical code that solves the continuity, momentum and energy conservation equation using FEM:

- Mass conservation: $\nabla \cdot u = 0$ 
    - $u$: velocity field 
- Momentum conservation: $\nabla \cdot \mathbf \tau + \nabla \cdot P = 0$
    - $\tau$ deviatoric stress: $\tau = 2 \eta_{\mathrm{eff}}\dot{\mathbf{\varepsilon}}$
        - $\dot{\mathbf{\varepsilon}}$ deviatoric strain tensor: $\dot{\mathbf{\varepsilon}} = \nabla u + \nabla u ^{T} $
        - $\eta_{\mathrm{eff}}$ effective viscosity: $\eta_{\mathrm{eff}} = f(T,P,\dot{\mathbf{\varepsilon}}_{II})$
            -  $\dot{\mathbf{\varepsilon}}_{II}$ deviatoric strain rate second invariant




---

## Installation

Clone the repository:

```bash
git clone https://github.com/username/project-name.git
cd project-name