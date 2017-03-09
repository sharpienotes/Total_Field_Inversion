#### MEDI (Morphology Enabled Dipole Inversion) method for QSM
 
#### paper source: 
Liu et al,: Morphology Enabled Dipole Inversion (MEDI) from
a Single-Angle Acquisition: Comparison with COSMOS in Human Brain Imaging

# Basics:
- Arbitrary susceptibility distribution from the measured MR signal phase: 
    - challenging, ill-conditioned inverse problem
    
### COSMOS: 
- can be used for this
- needs multiple orientation measuements though
- tough to do with people
- keeps full fidelity of data
- alternatives:
    - truncated k-space division
    - regularization encoding a priori information
    - single orientation calculation techniques:
        - variety of them exist
        - more practical
        - all subject to systematic bias 
            -  source: disagreement between: 
                - assumed mathematical properties 
                - physical reality
                    
### MEDI: 
- uses only a single acquisiton to get a similar result 
- here: sparsifying the edges in quantitative susceptibility map that do not have a corrsponding edge in the magnitude image
- incorporating morphological information already available in magnitude images
- successfully suppresses streaking artifacts 
- uses: location of 
    - edges in \chi distribution almost same as 
     - in magnitude images obtained in same acquisition
- sparse discordance
    - enforced by: weighted l1 minimization 
        - penalizes \chi at voxels not part of an interface in magnitude image
        - constrained by data fidelity
            - ensures agreement betw.:
                - local field induced by estimated \chi distribution 
                - local field as measured from the phase image
                
    ##### Susceptibility map: 
    - get knowledge about
        - iron 
        - calcium 
        - gadolinium
        
    ##### Nonferromagnetic biomaterial: 
    - generates local field:
    - z component (along B) equal to 
        - convolution of the volume susceptibility distribution  
        - and unit dipole field
            - magic angle: dipole field has zeros in Fourier space
            - creates noise amplification and ill-conditioned problem
            
    ##### Getting the field map:
    1) extract phase images from comples MRI data
    
    2) one-dimensional temporal unwrapping of the phase in each voxel
    
    3) weighted least-squares fit of temporally unwrapped phases 
        1) in each voxel 
        2) over TE
        
    4) frequency aliasing on field map:
        - magnitude map guided spatial unwrapping algorithm
        
    5) projection onto dipole fields procedure 
        1) removes background 
        2) all the voxels inside FOV but outside the brain region assumed responsible for background field inside brain
        3) strength of dipole in each background voxel 
            - weighted least-squares fit to field inside brain
            
    6) corrected field as input for field-to-source inverse problem
        
    ##### Susceptibility analysis:
     - described in detail how to compare the two methods and how to get the desired data from the scans 
       
#### Results

