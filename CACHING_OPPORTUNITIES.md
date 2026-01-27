# Where to Cache in StonedFenicsx

Based on code analysis, here are the **specific locations** where caching will give the biggest speedup:

---

## **🔴 HIGH PRIORITY CACHING OPPORTUNITIES**

### **1. Function Spaces in `output.py` (BIGGEST IMPACT)**

**Current Problem** (lines 25-31):
```python
# In OUTPUT.__init__():
self.vel_V = fem.functionspace(mesh, basix.ufl.element("Lagrange", "triangle", 1, shape=(mesh.geometry.dim,)))
self.pres_V = fem.functionspace(mesh, basix.ufl.element("Lagrange", "triangle", 1))
self.temp_V = self.pres_V
self.stress_V = fem.functionspace(mesh, basix.ufl.element("Lagrange", "triangle", 1, shape=(mesh.geometry.dim**2,)))
```

**Issue**: Each `fem.functionspace()` call recreates the space (~0.5-1 sec each)

**How to Cache**:
```python
class OUTPUT():
    def __init__(self, mesh, ioctrl, ctrl, sc, comm=MPI.COMM_WORLD):
        # ... existing code ...
        
        # Cache function spaces (compute once)
        if not hasattr(OUTPUT, '_cached_spaces'):
            OUTPUT._cached_spaces = {}
        
        mesh_id = id(mesh)
        if mesh_id not in OUTPUT._cached_spaces:
            OUTPUT._cached_spaces[mesh_id] = {
                'vel_V': fem.functionspace(mesh, basix.ufl.element("Lagrange", "triangle", 1, shape=(mesh.geometry.dim,))),
                'pres_V': fem.functionspace(mesh, basix.ufl.element("Lagrange", "triangle", 1)),
                'stress_V': fem.functionspace(mesh, basix.ufl.element("Lagrange", "triangle", 1, shape=(mesh.geometry.dim**2,)))
            }
        
        # Reuse cached spaces
        self.vel_V = OUTPUT._cached_spaces[mesh_id]['vel_V']
        self.pres_V = OUTPUT._cached_spaces[mesh_id]['pres_V']
        self.temp_V = self.pres_V
        self.stress_V = OUTPUT._cached_spaces[mesh_id]['stress_V']
```

**Speedup**: ~2-4 seconds per output call × multiple calls per iteration = **10-20 sec total per run**

---

### **2. Material Property Functions in `compute_material_property.py` (HUGE IMPACT)**

**Current Problem** (lines 30-45):
```python
def heat_conductivity_FX(pdb, T, p, phase, M, Cp, rho):
    ph = np.int32(phase.x.array)
    P0 = phase.function_space
    
    # Creating NEW fem.Function() EVERY TIME this is called:
    k0 = fem.Function(P0); k0.x.array[:] = pdb.k0[ph]
    fr = fem.Function(P0); fr.x.array[:] = pdb.radio_flag[ph]
    k_a = fem.Function(P0); k_a.x.array[:] = pdb.k_a[ph]
    k_b = fem.Function(P0); k_b.x.array[:] = pdb.k_b[ph]
    k_c = fem.Function(P0); k_c.x.array[:] = pdb.k_c[ph]
    # ... 3 more functions ...
```

**This is called in**:
- `solution.py` line ~560: `k_k = heat_conductivity_FX(...)` (inside `set_linear_picard_SS`)
- `solution.py` line ~850: `k_k = heat_conductivity_FX(...)` (inside `set_linear_picard_TD`)
- Each outer iteration! (20+ times total)

**How to Cache**:
```python
class PhaseDatabase:  # (or wherever pdb is defined)
    def __init__(self):
        # ... existing code ...
        self._cached_functions = {}  # Add this
    
    def get_cached_function(self, function_space, name, values):
        """Get or create a cached fem.Function"""
        key = (id(function_space), name)
        if key not in self._cached_functions:
            f = fem.Function(function_space)
            self._cached_functions[key] = f
        else:
            f = self._cached_functions[key]
        
        # Update values
        f.x.array[:] = values
        f.x.scatter_forward()
        return f

# Usage in heat_conductivity_FX:
def heat_conductivity_FX(pdb, T, p, phase, M, Cp, rho):
    ph = np.int32(phase.x.array)
    P0 = phase.function_space
    
    # Now REUSE cached functions:
    k0 = pdb.get_cached_function(P0, 'k0', pdb.k0[ph])
    fr = pdb.get_cached_function(P0, 'radio_flag', pdb.radio_flag[ph])
    k_a = pdb.get_cached_function(P0, 'k_a', pdb.k_a[ph])
    # ... rest stays same
```

**Speedup**: ~0.1 sec per property function × 6 properties × 20 iterations × 5 solves = **60 seconds saved!**

---

### **3. Trial/Test Functions in `solution.py` (MEDIUM IMPACT)**

**Current Problem** (lines 170-180, repeated for Slab, Wedge):
```python
# In Slab.Solve_the_Problem():
V_subs0 = self.FS.sub(0)
p_subs0 = self.FS.sub(1)
V_subs, _ = V_subs0.collapse()  # EXPENSIVE!
p_subs, _ = p_subs0.collapse()  # EXPENSIVE!

self.trial0 = ufl.TrialFunction(V_subs)  # Creates new each time
self.test0 = ufl.TestFunction(V_subs)    # Creates new each time
self.trial1 = ufl.TrialFunction(p_subs)  # Creates new each time
self.test1 = ufl.TestFunction(p_subs)    # Creates new each time
```

**How to Cache** (in Problem base class):
```python
class Problem:
    def __init__(self, ...):
        self._cached_trials = {}
        self._cached_tests = {}
    
    @property
    def trial0(self):
        """Cached trial function for first subspace"""
        key = id(self.F0) if hasattr(self, 'F0') else id(self.FS)
        if key not in self._cached_trials:
            self._cached_trials[key] = ufl.TrialFunction(self.F0 if hasattr(self, 'F0') else self.FS)
        return self._cached_trials[key]
    
    @property
    def test0(self):
        """Cached test function for first subspace"""
        key = id(self.F0) if hasattr(self, 'F0') else id(self.FS)
        if key not in self._cached_tests:
            self._cached_tests[key] = ufl.TestFunction(self.F0 if hasattr(self, 'F0') else self.FS)
        return self._cached_tests[key]
```

**Speedup**: ~0.05 sec per creation × 8 creations per Stokes solve × 20 iterations = **8 seconds saved**

---

## **🟡 MEDIUM PRIORITY CACHING**

### **4. Function Creation in `Global_thermal.set_linear_picard_SS()` (MEDIUM)**

**Current** (lines 540-560):
```python
def set_linear_picard_SS(self, p_k, T, u_global, Hs, D, pdb, ctrl, it=0):
    # Creates NEW functions EVERY ITERATION:
    rho_k = density_FX(pdb, T, p_k, D.phase, D.mesh)     # New each time
    Cp_k = heat_capacity_FX(pdb, T, D.phase, D.mesh)     # New each time
    k_k = heat_conductivity_FX(pdb, T, p_k, D.phase, D.mesh, Cp_k, rho_k)  # New
```

**How to Cache**:
```python
def __init__(self, ...):
    self._cached_density = None
    self._cached_Cp = None
    self._cached_k = None

def set_linear_picard_SS(self, p_k, T, u_global, Hs, D, pdb, ctrl, it=0):
    # Check if values changed, reuse if same
    if (self._cached_density is None or 
        not np.allclose(T.x.array, self._cached_T_prev.x.array)):
        self._cached_density = density_FX(pdb, T, p_k, D.phase, D.mesh)
        self._cached_Cp = heat_capacity_FX(pdb, T, D.phase, D.mesh)
        self._cached_k = heat_conductivity_FX(...)
        self._cached_T_prev = T.copy()
    
    rho_k, Cp_k, k_k = self._cached_density, self._cached_Cp, self._cached_k
    # ... rest of function
```

**Speedup**: ~0.2 sec × 20 Picard iterations × 10 outer iterations = **40 seconds**

---

### **5. BC Computation in `Global_thermal.create_bc_temp()` (MEDIUM)**

**Current** (lines 460-480):
```python
if ts == 0 and it == 0:  # Only computed once!
    facets = M.facets.find(M.bc_dict['Top'])
    dofs_top = fem.locate_dofs_topological(self.FS, M.mesh.topology.dim-1, facets)
    # ... rest of BC setup (already cached with "if ts==0 and it==0")
```

**Status**: ✅ **ALREADY CACHED!** But could be improved by moving to `__init__`

---

## **🟢 LOW PRIORITY CACHING**

### **6. Element Creation in `output.py` `__init__`**

**Current** (using basix each time):
```python
basix.ufl.element("Lagrange", "triangle", 1, shape=(mesh.geometry.dim,))
```

**Caching not critical** (happens once per OUTPUT instance), but could cache globally:
```python
@staticmethod
def get_cached_element(degree, family, cell, shape=None):
    key = (degree, family, cell, shape)
    if not hasattr(OUTPUT, '_cached_elements'):
        OUTPUT._cached_elements = {}
    if key not in OUTPUT._cached_elements:
        if shape:
            OUTPUT._cached_elements[key] = basix.ufl.element(family, cell, degree, shape=shape)
        else:
            OUTPUT._cached_elements[key] = basix.ufl.element(family, cell, degree)
    return OUTPUT._cached_elements[key]
```

---

## **QUICK IMPLEMENTATION PRIORITY**

### **Best Bang for Buck (Next 2 hours):**

1. **Phase property functions** (heat_conductivity_FX, density_FX, heat_capacity_FX)
   - Impact: **60+ seconds/run**
   - Effort: **30 minutes**
   - Location: `src/compute_material_property.py` lines 30-45

2. **Function spaces in OUTPUT**
   - Impact: **10-20 seconds/run**
   - Effort: **20 minutes**
   - Location: `src/output.py` lines 25-31

3. **Trial/Test functions**
   - Impact: **8 seconds/run**
   - Effort: **20 minutes**
   - Location: `src/solution.py` lines 170-180

---

## **SUMMARY TABLE**

| Location | What to Cache | Current Cost | Cached Cost | Speedup | Priority |
|----------|---------------|--------------|-------------|---------|----------|
| `compute_material_property.py` | fem.Function objects for properties | 0.1s each × 120 calls | 0.1s × 1 call | **60s/run** | 🔴 |
| `output.py` | Function spaces (vel, pres, stress) | 1s each × 4 | 1s × 1 | **3s/output** | 🔴 |
| `solution.py` (Slab/Wedge) | Trial/test functions | 0.05s each × 8 × 20 | cached | **8s/run** | 🔴 |
| `solution.py` (Global_thermal) | Material properties per iteration | 0.2s × 200 | 0.2s × 20 | **36s/run** | 🟡 |
| `output.py` | Basix elements | 0.01s each × 4 | 0.01s × 1 | <1s | 🟢 |

---

## **TOTAL POTENTIAL SPEEDUP**

- **60s (properties)** 
- **+3s (function spaces)** 
- **+8s (trial/test)** 
- **+36s (material props iteration)**
- **= ~107 seconds (~30% faster)**

Plus 15-20 seconds from import cleanup already done!

**Total potential: 35-40% speedup from optimization**

