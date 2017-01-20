import scipy as sp
from scipy import stats

def get_sample_D(n, m, num_sim, r2=0.9):
    D_avg = sp.zeros((m, m), dtype='float32')
    for sim_i in range(num_sim):
        # Generating correlated training genotypes
        X = sp.empty((m, n), dtype='float32')
        X[0] = stats.norm.rvs(size=n)        
        for j in range(1, m):
            X[j] = sp.sqrt(r2) * X[j - 1] + sp.sqrt(1 - r2) * stats.norm.rvs(size=n)
        
        X = sp.mat(X).T
        # Make sure that they have 0 mean and variance 1. (Normalizing)
        X_means = X - sp.mean(X, 0)
        X = X_means / sp.std(X_means, 0) 
        # Calculating the marker correlation matrix (if needed)

        D_avg += X.T * X / n
    D_avg = D_avg / num_sim
    return D_avg



def simulate_genotypes(n, m, n_samples):
	val_set = []
	for i in xrange(n_samples):
		snps = stats.norm.rvs(size=(m, n))
		snps_means = sp.mean(snps, 1)
		snps_stds = sp.std(snps, 1)
		snps_means.shape = (m, 1)
		snps_stds.shape = (m, 1)
		snps = (snps - snps_means) / snps_stds
		val_set.append(snps)
	return val_set

def simulate_genotypes_w_ld(n, m, n_samples, m_ld_chunk_size=100, r2=0.9, validation = False):
	val_set_gen = []
	val_set_D = []
	for i in xrange(n_samples):
		snps = sp.zeros((m, n), dtype='float32')
		num_chunks = m / m_ld_chunk_size
		for chunk in xrange(num_chunks):
			X = sp.zeros((m_ld_chunk_size, n), dtype='float32')
			X[0] = stats.norm.rvs(size=n)
			for j in xrange(1, m_ld_chunk_size):
				X[j] = sp.sqrt(r2) * X[j - 1] + sp.sqrt(1 - r2) * stats.norm.rvs(size=n)
			start_i = chunk * m_ld_chunk_size
			stop_i = start_i + m_ld_chunk_size
			snps[start_i:stop_i] = X
		snps_means = sp.mean(snps, axis=1)
		snps_stds = sp.std(snps, axis=1)
		snps_means.shape = (m, 1)
		snps_stds.shape = (m, 1)
		snps = (snps - snps_means) / snps_stds
		val_set_gen.append(snps)
		if validation:
			val_set_D = 0
		else:
			val_set_D.append(get_sample_D(n=n, m=m, num_sim=100, r2=r2))
		if sp.isnan(val_set_gen).any():
			return simulate_genotypes_w_ld(n, m, n_samples, m_ld_chunk_size=100, r2=0.9)
	return val_set_gen, val_set_D





def write_genotypes(filename, genotype):
	sp.savetxt(filename + ".geno.txt", genotype[0])
	sp.savetxt(filename + ".D.txt", genotype[1])
