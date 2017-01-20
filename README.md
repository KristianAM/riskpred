# riskpred
Python script for simulating genotypes, phenotypes and subsequent risk prediction.
Contains implementations of LDpred (Vilhjálmsson et. al. 2015) and COJO (Yang et. al. 2012).
A combination of LDpred and COJO is implemented called COJOpred.

# genotypes.py
Contains functions for simulating genotypes, with and without LD, and simulating reference LD matrix.

# simulate_genotypes(n, m, n_samples)
Function for simulating genotypes without LD  
Set n and m to be the desired sample size and number of SNPs respectively. n_samples is the number of genotypes simulated.  
Returns a list of n X m genotypes.  

# simulate_genotypes_w_ld(n, m, n_samples, m_ld_chunk_size=100, r2=0.9, validation = False)  
Function for simulating genotypes with LD  
Set n and m to be the desired sample size and number of SNPs respectively. n_samples is the number of genotypes simulated.  
m_ld_chunk_size is the maximum LD radius. All simulated SNPs are in LD with other SNPs within this radius, at levels following a bell curve maxing out at r2. Validation = True if no reference LD simulation is required. Speeds up the computation time.  

# get_sample_D(n, m, num_sim, r2=0.9)
Function for simulating reference LD matrix.
n and m is the same as other functions, num_sim is the number of iterations, the more iterations the more precise the estimate. 100 is an adequate number of simulations. r2 is the maximum level of LD.

# test_acc.py
Contains functions for simulating phenotypes and testing various hypotheses.

# test_accuracy(n, m, n_samples=10, genotype=None, h2=0.5, p=1.0, r2=0.9, m_ld_chunk_size=100, p_threshold=5e-8, validation_set=None, alpha=None, verbose=False, variance = False)
Main function, n, m and n_samples is as in genotypes.py.  
If no genotype is given, it will automatically simulate some itself. h2 = 0.5 is the heritability desired, p=1.0 is the level of complexity, i.e. the proportion of causal SNPs. r2 and m_ld_chunk_size is the same as in the case of genotypes.py.  
p_threshold is the desired pvalue threshold, default is the standard genome-wide significance threshold.  
alpha is the desired weighting parameter between the two parts in the COJOpred implementation. if None, it will estimate alpha using a cross-validation scheme. If variance is true, it will instead use variance explained by cojo as weighting parameter.  
returns a list of r2 accuracy estimates for each function on the validation set.

# example
test_accuracy(n = 3000, m = 8000, n_samples=10, genotype=None, h2=0.5, p=0.03, r2=0.9, m_ld_chunk_size=100, p_threshold=5e-8, validation_set=None, alpha=0.5, verbose=False, variance = True)  
Will simulate 10 traits with 8000 SNPs of which 3% of them are causal. The sample size used to train the models will be 3000. This will return a list of traits, each containing a list of the r2 accuracies for each method.
