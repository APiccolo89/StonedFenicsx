# Examples 

## Benchmarks
**StonedFEniCSx** has been tested against the benchmarks of {cite}`van2008community`. The benchmarks are used also for the testing routines. In the table below, there are the ranges of the benchmarks and the results of **StonedFEniCSx**. 


| Case | $T_{11,11}$ Range | $T_{11,11}$ Mean | $\|T_{\mathrm{slab}}\|$ Range | $\|T_{\mathrm{slab}}\|$ Mean | $\|T_{\mathrm{wedge}}\|$ Range | $\|T_{\mathrm{wedge}}\|$ Mean | $T_{11,11}$ (This work) | $\|T_{\mathrm{slab}}\|$ (This work) | $\|T_{\mathrm{wedge}}\|$ (This work) |
|------|------|------|------|------|------|------|------|------|------|
| 1c | 387.78--397.55 | 390.47 | 488.00--511.09 | 502.58 | 847.70--854.99 | 851.88 | 389.80 | 505.50 | 855.70 |
| 2a | 570.30--584.20 | 579.09 | 592.80--614.09 | 605.95 | 1000.00--1007.31 | 1003.14 | 573.13 | 602.90 | 1001.1 |
| 2b | 550.17--585.70 | 573.06 | 591.30--608.85 | 601.72 | 984.08--1000.05 | 995.54 | 576.20 | 600.36 | 997.40 |

### Effects of non-linearties on the benchmark values 
 
(fig:f1_example_benchmark)=
```{figure} images_doc/Benchmark.png
:width: 500px
```{figure} images_doc/Benchmark.png
:name: fig:f1_example_benchmark
:width: 500px

**Figure 1.** [a]: **Case 2b** steady-state temperature field; [b]: **case 2b non-linear and crustal unit** steady-state temperature field. [c-d]: Convergence rate vs number of iterations. The *green lines* represent the mass, momentum, and energy conservation relative residuum; *red lines* represent the relative combined difference of the solution as a function of iteration.
```

The case `2b` has been repeated with non-linearties, and with crust-units both in the overriding and subducting plate. 

- **case 2b non-linear**:  $T_{11,11}$ = 558.5840 degC, $\|T_{\mathrm{slab}}\|$= 609.5550 degC,  $\|T_{\mathrm{wedge}}\|$ = 942.9349
- **case 2b non-linear and crustal unit**:  $T_{11,11}$ = 599.8287 degC, $\|T_{\mathrm{slab}}\|$= 635.9449 degC,  $\|T_{\mathrm{wedge}}\|$ = 962.3271

In {numref}`fig:f1_example_benchmark`, two representative benchmarks are shown (**Case 2b** and **case 2b non-linear and crustal unit**). The figure have been produced with supplementary scripts that will be released together with the package. 


## Sensitivity study 

## Mexico example slab 

## Time Dependent 

