# TODO — StonedFenicsx

## Fase 1 — Stabilizzare (codice non funzionante)

- [ ] Completare `stonedfenicsx/solver_module/solution_routine.py` — `outerloop_operation()` troncata, manca il corpo del loop
- [ ] Completare `stonedfenicsx/solver_module/problems_solution.py` — mancano le classi `Global_thermal`, `Global_pressure`, `Wedge`, `Slab`
- [ ] Decidere cosa fare con `stonedfenicsx/right_boundary.py`:
  - se in sviluppo → spostare su branch separato
  - se rotto → rimuovere da main
  - problemi specifici: `res` non inizializzato, `phases` non definito, `Assembly_matrix()`/`Assembly_vector()` non definite, assignment incompleto `Pdb =`
- [ ] Correggere variabile `t` non definita in `update_age_lhs()` — `thermal_structure_ocean.py`

## Fase 2 — Pulizia

- [ ] Rimuovere campi marcati `# REMOVE` da `numerical_control.py` (`van_keken`, `phase_wz`, `adiabatic_heating`)
- [ ] Aggiungere max-iterations al while loop in `right_boundary.py` (rischio loop infinito)
- [ ] Aggiungere check `np.isinf()` in `solver_utilities.py` — `compute_residuum_outer()` controlla solo `np.isnan()`
- [ ] Rimuovere blocchi di plotting commentati da `read_slab_surface.py` (linee ~62-104)
- [ ] Rimuovere codice commentato morto da `Melting_parametrisation.py` e `phase_db.py`

## Fase 3 — Leggibilita'

- [ ] Rinominare `domainA/B/C/G` con nomi semantici (`slab_domain`, `wedge_domain`, ecc.)
- [ ] Popolare `stonedfenicsx/__init__.py` con le export principali
- [ ] Aggiornare `README.md`: cosa fa il codice, come si installa, come si lancia un esempio

## Fase 4 — Test

- [ ] Aggiungere un test end-to-end su un caso minimo (verifica che il solver giri senza esplodere)
- [ ] Correggere tolleranze in `test_benchmark_subdomains.py` — `np.isclose(..., 5e-1, 1e-1)` non verifica nulla
- [ ] Aggiungere il file YAML di input al repo (o generarlo nel test setup)
- [ ] Sostituire i valori hardcoded di nodi/celle in `test_mesh_generation.py` con checks parametrici

## Sperimentazione (refactoring architetturale)

- [ ] Prototipare classi con metodi che ricevono funzioni configurate come input
  - candidati naturali: condizioni al contorno, leggi reologiche, struttura termica iniziale
  - esempio: `ThermalProblem(mesh, conductivity_fn, heat_source_fn, bcs)`
  - iniziare da un modulo solo, non riscrivere tutto
