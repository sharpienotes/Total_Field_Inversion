#### MEDI (Morphology Enabled Dipole Inversion) method for QSM
 
#### paper source: 
Liu et al,: Morphology Enabled Dipole Inversion (MEDI) from
a Single-Angle Acquisition: Comparison with COSMOS in Human Brain Imaging

### Basics:
- arbitrary susceptibility distribution from the measured MR signal phase: 
    - challenging, ill-conditioned inverse problem
- COSMOS: can be used for this
    - needs multiple orientation measuements though
    - tough to do with people
- MEDI: uses only a single acquisiton to get a similar result 
    - here: sparsifying the edges in quantitative susceptibility map that do not have a corre- sponding edge in the magnitude image
- susceptibility map: get knowledge about
    - iron 
    - calcium 
    - gadolinium
- nonferromagnetic biomaterial generates local field:
    - z component (along B) equal to 
        - convolution of the volume susceptibility distribution  
        - and unit dipole field

