import scipy as sp
from scipy import stats

def get_sample_D(n, m, num_sim, r2=0.9):
    D_avg = sp.zeros((m, m), dtype='single')
    for sim_i in range(num_sim):
        # Generating correlated training genotypes
        X = sp.empty((m, n), dtype='single')
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

def simulate_genotypes_w_ld(n, m, n_samples, m_ld_chunk_size=100, r2=0.9):
	val_set_gen = []
	val_set_D = []
	for i in xrange(n_samples):
		snps = sp.zeros((m, n), dtype='single')
		num_chunks = m / m_ld_chunk_size
		for chunk in xrange(num_chunks):
			X = sp.empty((m_ld_chunk_size, n), dtype='single')
			X[0] = stats.norm.rvs(size=n)
			for j in xrange(1, m_ld_chunk_size):
				X[j] = sp.sqrt(r2) * X[j - i] + (1 - sp.sqrt(r2)) * stats.norm.rvs(size=n)
			start_i = chunk * m_ld_chunk_size
			stop_i = start_i + m_ld_chunk_size
			snps[start_i:stop_i] = X
		snps_means = sp.mean(snps, axis=1)
		snps_stds = sp.std(snps, axis=1)
		snps_means.shape = (m, 1)
		snps_stds.shape = (m, 1)
		snps = (snps - snps_means) / snps_stds
		val_set_gen.append(snps)
		val_set_D.append(get_sample_D(n=n, m=m_ld_chunk_size, num_sim=100, r2=r2))
		if sp.isnan(val_set_gen).any():
			return simulate_genotypes_w_ld(n, m, n_samples, m_ld_chunk_size=100, r2=0.9)
	return val_set_gen, val_set_D





def write_genotypes(filename, genotype):
	sp.savetxt(filename + ".geno.txt", genotype[0])
	sp.savetxt(filename + ".D.txt", genotype[1])

def printtable(filename, p, N, M, Ntraits, validation_size=1000, validationN=10):
	with open(filename, 'w') as f:
		print >> f, 'N \t M \t P \t SMT \t PRS \n'
		for i in range(len(N)):
			print "N"
			print N[i]
			for m in range(len(M)):
				print "M"
				print M[m]
				validation = simulate_genotypes(n=validation_size, m=M[m], n_samples=validationN)
				for j in p:
					print j
					output = simulate_phenotypes_fast(validation, n=N[i], m=M[m], num_traits=n_traits, p=j)
					for l in range(n_traits):
						print >> f, N[i], "\t", M[m], "\t", j, "\t", output[0][l], "\t", output[1][l], "\n"


if __name__ == "__main__":
	# validation = simulate_genotypes(n = 1000, m = 10000, n_samples = 1)
	# test = simulate_phenotypes_fast(validation, n = 10000, m = 10000, num_traits = 1, p = 0.05)
	# x = simulate_genotypes_w_ld(n = 1000, m = 5000, n_samples = 2, m_ld_chunk_size = 100, r2 = 0.9)
	#### construct test files
	# N = [10,50, 100, 500, 1000,5000,10000, 50000, 100000]
	# m = 6000
	# r2 = [0.5,0.6,0.7,0.8,0.9]
	# for n in N:
	# 	for r in r2:
	# 		print n,m,r
	# 		data = simulate_genotypes_w_ld(n = n, m = m, n_samples = 1, m_ld_chunk_size = 100, r2 = r)
	# 		filename = "genotypes/genotypes.n." + str(n) + "m." + str(m) + "r2." + str(r)
	# 		write_genotypes(filename, data[0])
	""
