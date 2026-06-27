1# Thermal Boundary Module — Call Flow

```mermaid
flowchart TD

    %% ── Entry point ──────────────────────────────────────────────
    CBC["configure_boundary_condition\n(public entry point)"]

    CBC -->|left_right=True|  CTL["configure_thermal_bc\nleft boundary"]
    CBC -->|left_right=False| CTR["configure_thermal_bc\nright boundary"]

    %% ── Branch: recalculate or read cache ────────────────────────
    CTL --> REC{recalculate?}
    CTR --> REC

    REC -->|Yes|  CTB["compute_thermal_boundary\n@timing_function"]
    REC -->|No|   RTF["read_temporary_file"]

    %% ── Cache read path ──────────────────────────────────────────
    RTF --> FEXIST{H5 file\nexists?}
    FEXIST -->|No|  CTB
    FEXIST -->|Yes| CMP["check_material_property\nvalidates pdb vs cached coefficients"]
    CMP --> LOAD["load temperature_1d / z\nfrom HDF5 → ctrl_tbc"]

    %% ── Compute path ─────────────────────────────────────────────
    CTB --> IGD["initialise_geometry_1d\nbuild z vector + phase array"]
    IGD --> FPP["fill_phase_properties\nassign phase id per depth"]

    CTB --> VK{van_keken\n& left BC?}
    VK -->|Yes| HSC["compute_half_space_cooling_model_analytical\nerf half-space solution"]
    VK -->|No|  SOL["solve_temperature_1d_bc\ntime-stepping loop"]

    %% ── Time-stepping internals ──────────────────────────────────
    SOL --> LIT["_compute_lithostatic_pressure\n@njit  Picard pressure integration"]
    SOL --> BCM["build_coefficient_matrix\n@njit  assemble CN system"]
    BCM --> CPK["compute_cp_k_rho\n@njit  evaluate ρ, Cp, k per node"]
    CPK --> CTP["compute_thermal_properties\n@njit  (phase_db)"]
    BCM --> DRH["density / heat_capacity\n@njit  (phase_db)"]

    %% ── Save path ────────────────────────────────────────────────
    CTB --> SAVE{save_data\n& rank==0?}
    SAVE -->|Yes| RCC["check_race_condition\npsutil file-lock check"]
    SAVE -->|Yes| SDS["save_data_set\nwrite / overwrite HDF5 dataset"]

    %% ── Styling ──────────────────────────────────────────────────
    classDef entry    fill:#2d6a4f,color:#fff,stroke:#1b4332
    classDef njit     fill:#1d3557,color:#fff,stroke:#0d1b2a
    classDef io       fill:#6d4c41,color:#fff,stroke:#3e2723
    classDef decision fill:#f4a261,color:#000,stroke:#e76f51

    class CBC entry
    class LIT,BCM,CPK,CTP,DRH njit
    class RCC,SDS,RTF,CMP,LOAD io
    class REC,VK,FEXIST,SAVE decision
```

## Function index

| Function | Type | Role |
|---|---|---|
| `configure_boundary_condition` | public | Runs left then right BC setup |
| `configure_thermal_bc` | public | Dispatch: recompute or read cache |
| `read_temporary_file` | internal | Load BC from HDF5 cache |
| `check_material_property` | internal | Validate cached pdb matches current one |
| `compute_thermal_boundary` | internal `@timing` | Orchestrates geometry + solver + save |
| `initialise_geometry_1d` | internal | Build z grid and phase array |
| `fill_phase_properties` | internal | Assign phase id per depth level |
| `compute_half_space_cooling_model_analytical` | internal | Analytical erf solution (benchmark) |
| `solve_temperature_1d_bc` | internal | Crank-Nicolson + Picard time loop |
| `build_coefficient_matrix` | `@njit` | Assemble FD matrix and RHS |
| `compute_cp_k_rho` | `@njit` | Evaluate ρ, Cp, k over 1D column |
| `_compute_lithostatic_pressure` | `@njit` | Picard pressure integration |
| `check_race_condition` | utility | psutil file-lock guard |
| `save_data_set` | utility | Write/overwrite single HDF5 dataset |
