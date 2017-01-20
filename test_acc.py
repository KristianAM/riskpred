import scipy as sp
from scipy import stats
import genotypes
from scipy import linalg
import LDpred


def get_LDpred_ld_tables(snps, ld_radius=100, ld_window_size=0, h2=None, n_training=None):
	"""
	Calculates LD tables, and the LD score in one go...
	"""

	ld_dict = {}
	m = len(snps)
	n = len(snps[0])
	ld_scores = sp.ones(m)
	for snp_i, snp in enumerate(snps):
		# Calculate D
		start_i = max(0, snp_i - ld_radius)
		stop_i = min(m, snp_i + ld_radius + 1)
		X = snps[start_i: stop_i]
		D_i = sp.dot(snp, X.T) / n
		r2s = D_i ** 2
		ld_dict[snp_i] = D_i
		lds_i = sp.sum(r2s - (1 - r2s) / (n - 2), dtype='float32')
		# lds_i = sp.sum(r2s - (1-r2s)*empirical_null_r2)
		ld_scores[snp_i] = lds_i
	ret_dict = {'ld_dict':ld_dict, 'ld_scores':ld_scores}

	if ld_window_size > 0:
		ref_ld_matrices = []
		inf_shrink_matrices = []
		for i, wi in enumerate(range(0, m, ld_window_size)):
			start_i = wi
			stop_i = min(m, wi + ld_window_size)
			curr_window_size = stop_i - start_i
			X = snps[start_i: stop_i]
			D = sp.dot(X, X.T) / n
			ref_ld_matrices.append(D)
			if h2 != None and n_training != None:
				A = ((m / h2) * sp.eye(curr_window_size) + (n_training / (1)) * D)
				A_inv = linalg.pinv(A)
				inf_shrink_matrices.append(A_inv)
		ret_dict['ref_ld_matrices'] = ref_ld_matrices
		if h2 != None and n_training != None:
			ret_dict['inf_shrink_matrices'] = inf_shrink_matrices
	return ret_dict


def simulate_traits_fast(n, m, h2=0.5, p=1.0):
	# simulate betas from a normal distribution with mean 0 and variance = h2/m.
	# p chooses architecture where 1 is infinitisimal, 0<p<1 is non-infinitisimal and p = 1/m is mendelian
	if p == 1.0:
		M = m
		betas = stats.norm.rvs(0, sp.sqrt(h2 / m), size=m)
	else:
		M = int(round(m * p))
		if M == 0:
			M = 1
		betas = sp.concatenate((stats.norm.rvs(0, sp.sqrt(h2 / M), size=M), sp.zeros(m - M, dtype=float)))
		sp.random.shuffle(betas)
	# coefficients are scaled to have mean 0 and variance 1

	betas_var = sp.var(betas)
	betas_scalar = sp.sqrt(h2 / (m * betas_var))
	betas = betas * betas_scalar
	# betas_list.append(betas)

	return betas

def LDinf(beta_hats, ld_radius, LD_matrix, n, h2):
	m = len(beta_hats)
	betainf = sp.zeros(m)

	for m_i in range(0, m, ld_radius):
		# calculate the ld corrected betaHats under the infinitesimal model
		
		m_end = min(m_i + ld_radius, m)
		if LD_matrix.shape == (ld_radius, ld_radius):
			A = ((m / h2) * sp.eye(min(ld_radius, m_end - m_i)) + (n / (1)) * LD_matrix)
		else:
			A = ((m / h2) * sp.eye(min(ld_radius, m_end - m_i)) + (n / (1)) * LD_matrix[m_i:m_end, m_i:m_end])
		try:
			Ainv = linalg.pinv(A)
		except Exception(err_str):
			print err_str
			print linalg.pinv(LD_matrix)
		betainf[m_i:m_end] = sp.dot(Ainv * n, beta_hats[m_i:m_end])
	return betainf


def ml_iter(beta_hats, snps, ld_radius, h2, p_threshold, printout = False, LD = None, LDmethod = True, p = 0.05, ref_ld = True, r2 = 0.9):
	var_explained_list = []
	var_explained_index = []
	non_selected_indices = []
	snps = sp.array(snps)
	# Ordering the beta_hats, and storing the order
	m = len(beta_hats)
	beta_hats = beta_hats.tolist()
	ld_table = {}
	for i in range(m):
		ld_table[i] = {'ld_partners':[i], 'beta_hat_list':[beta_hats[i]], 'sumsq' : beta_hats[i] ** 2,
						'pval' : 0, 'D' : sp.array([1.0]), 'D_inv':sp.array([1.0])}
		ld_table[i]['pval'] = stats.chi2.sf((ld_table[i]['beta_hat_list'][0] ** 2) * snps.shape[0], 1)
	n_test = len(snps[0])

	selected_indices = set()
	updated_beta_hats = beta_hats[:]
	updated_pvalues = [ld_table[i]['pval'] for i in range(m)]

	is_not_finished = True
	while is_not_finished:

		# Sort and select beta
		l = zip(updated_pvalues, range(m))
		l.sort()
		for pvalue, beta_i in l:
			if not beta_i in selected_indices:
				if pvalue <= p_threshold:
					selected_indices.add(beta_i)
					break
				else:
					is_not_finished = False
					break

		if is_not_finished:
			# Iterating over the window around the selected beta
			# Identy LD chunk
			print beta_i
			start_i = max(0, beta_i - ld_radius)
			end_i = min(beta_i + ld_radius, m)
			for i in range(start_i, end_i):
				if i == beta_i:
					continue
				else:
					# Update the LD matrix
					d = ld_table[i]
					ld_partners = d['ld_partners']
					beta_hat_list = d['beta_hat_list']
					ld_partners.append(beta_i)
					beta_hat_list.append(beta_hats[beta_i])
					#print "ldpartners2", ld_partners
					#print "betahatlist", beta_hat_list
					bs = sp.array(beta_hat_list, dtype='single')
					X = snps[ld_partners]
					if LD.shape == (m,m):
						D = LD[ld_partners]
					else:
						assert 0 > 1
					ld_table['D'] = D
					D_inv = linalg.pinv(D)
					ld_table['D_inv'] = D_inv
					
					updated_beta = sp.dot(D_inv, bs)
					sumsq2 = sp.sum(updated_beta ** 2)
					#print "updated beta", updated_beta
					#print "sumsq", d['sumsq']
					#print "sumsq2", sumsq2
					added_var_explained = sumsq2 - d['sumsq']
					# Is the added variance explained usually (always) larger than 0.  Perhaps a graph? 
					# How good is this approximation?
					var_explained_list.append(added_var_explained)
					var_explained_index.append(i)
					d['pval'] = stats.chi2.sf(added_var_explained, 1)
					d['sumsq'] = sumsq2
					updated_beta_hats[i] = updated_beta[0]
					updated_pvalues[i] = d['pval']

	cojo_updated_beta_hats = updated_beta_hats[:]
	selected_indices = list(selected_indices)
	non_selected_indices = []

	for i in range(m):
		if not i in selected_indices:
			d = ld_table[i]
			non_selected_indices.append(i)
			ld_partners = d['ld_partners'][1:]
			cojo_updated_beta_hats[i] = 0
			if len(ld_partners) >= 1:
				Xi = snps[i]
				Xpartners = snps[ld_partners]
				upb = sp.array(updated_beta_hats, dtype='single')[ld_partners]
				Di = LD[i, ld_partners]
				updated_beta_hats[i] = beta_hats[i] - sp.sum(sp.dot(Di, upb))

	# Remove cojosnps
	snps = sp.array(snps)
	newX = snps[non_selected_indices]
	assert newX.shape[0] + len(selected_indices) == len(beta_hats)
	newbetas = sp.array(updated_beta_hats)[non_selected_indices]

	m, n = newX.shape
	D = sp.dot(newX, newX.T) / float(n)

	betainf = LDinf(beta_hats=newbetas, ld_radius=ld_radius, LD_matrix=D, n=n, h2=h2)		
	if LDmethod:
		if ref_ld:
			ldDict = get_LDpred_ld_tables(genotypes.get_sample_D(50, m, num_sim = 50, r2=r2), ld_radius=ld_radius, h2=h2, n_training=n)
		else:
			ldDict = get_LDpred_ld_tables(newX, ld_radius=ld_radius, h2=h2, n_training=n)	
		betainf = LDpred.ldpred_gibbs(newbetas, start_betas=betainf, n=n, ld_radius=ld_radius, p=p, ld_dict=ldDict["ld_dict"], h2=h2)["betas"]


	betainf_with_0 = sp.zeros(len(beta_hats))
# 	updated_beta_hats_with_0 = sp.zeros(len(beta_hats))
	betainf_with_0[non_selected_indices] = betainf

# 	updated_beta_hats_with_0 = cojo_updated_beta_hats
	cojo_updated_beta_hats = sp.array(cojo_updated_beta_hats)
	if printout:
		with open("added_var_explained.csv", 'w') as f:
			print >>f, "added var explained \t index \n"
			for i in var_explained_list:
				print >>f, var_explained_list, "\t", var_explained_index, "\n"
	return cojo_updated_beta_hats, betainf_with_0, len(selected_indices)

def simulate_phenotypes(genotype, n, m, h2, p):
	betas = simulate_traits_fast(n, m, h2, p)
	phen_noise = stats.norm.rvs(0, sp.sqrt(1.0 - h2), size=n) 
	phen_noise = sp.sqrt((1.0 - h2) / sp.var(phen_noise)) * phen_noise
	genetic_part = sp.dot(genotype.T, betas)
	genetic_part = sp.sqrt(h2 / sp.var(genetic_part)) * genetic_part
	train_phen = genetic_part + phen_noise
	train_phen = (1 / sp.std(train_phen)) * train_phen
	return train_phen, betas

# def simulate_beta_hats(true_betas, n, m, r2, LD_matrix, m_ld_chunk_size):
# 	noises = stats.norm.rvs(0, 1, size=m)
# 	if r2 == 0:
# 		beta_hats = betas + sp.sqrt(1.0 / n) * noises
# 	else:
# 		# if ld
# 		C = sp.sqrt(((1.0) / n)) * linalg.cholesky(LD_matrix)
# 		D_I = linalg.pinv(LD_matrix)
# 		betas_ld = sp.zeros(m)
# 		noises_ld = sp.zeros(m)
# 		for m_i in range(0, m, m_ld_chunk_size):
# 			m_end = m_i + m_ld_chunk_size
# 			betas_ld[m_i:m_end] = sp.dot(LD_matrix, true_betas[m_i:m_end])
# 			noises_ld[m_i:m_end] = sp.dot(C.T, noises[m_i:m_end])
# 		beta_hats = betas_ld + noises_ld
# 	return beta_hats



def test_accuracy(n, m, n_samples=10, genotype=None, h2=0.5, p=1.0, r2=0.9, m_ld_chunk_size=100, p_threshold=5e-8, validation_set=None, alpha=None, verbose=False, variance = False, var_printout = False, ref_ld = False):
	# Simulate training data
	if verbose == True:
		print "simulating training set"
	training_set = genotypes.simulate_genotypes_w_ld(n=n, m=m, n_samples=1, m_ld_chunk_size=m_ld_chunk_size, r2=r2)
	traingeno = training_set[0][0]
	sample_Dref = genotypes.get_sample_D(10, m, num_sim = 50, r2=r2)
	train_phen, train_betas = simulate_phenotypes(genotype=traingeno, n=n, m=m, h2=h2, p=p)

	# estimate marginal beta hats
	betas_marg = (1.0 / n) * sp.dot(train_phen, traingeno.T)

	# Estimate or import alpha hyperparameter
	if verbose == True:
		print "estimating alpha"
	if alpha == None:
		# Estimate the alpha for the training data that you already had, 
		alpha = estimate_alpha(train_set = traingeno, train_phen = train_phen, betas_marg = betas_marg, train_betas = train_betas, n=n, m=m, h2=h2, p=p, r2=r2, m_ld_chunk_size=m_ld_chunk_size, p_threshold=p_threshold, n_samples = 1)

	# simulate validation data or import data
	if genotype == None:
		if r2 == 0:
			geno = genotypes.simulate_genotypes(n=2000, m=m, n_samples=n_samples)
		else:
			sample = genotypes.simulate_genotypes_w_ld(n=2000, m=m, n_samples=n_samples, m_ld_chunk_size=m_ld_chunk_size, r2=r2)
			geno = sample[0]
	else:
		if r2 == 0:
			geno = genotype[0]
		else:
			geno = genotype[0]

	# phenotype lists
	true_phen = []
	YhatsSMT = []
	YhatsPRS = []
	YhatsPval = []
	Yhatsinf = []
	YhatsLDpred = []
	Yhatscojo = []
	Yhatscojopred = []
	Yhatscojopred_betainf = []
	Yhatscojopred_concatenated = []
	YhatsLDpredref = []
	Yhatscojopred3 = []
	Yhatscojopred3_betainf = []
	Yhatscojopred3ref = []
	Yhatscojopred3ref_betainf = []
	Yhatscojopred3_concatenated = []
	Yhatscojopred3ref_concatenated = []
	Yhatscojoref = []
	Yhatsinfref = []
	# validation accuracy list
	val_accuracy_SMT = []
	val_accuracy_PRS = []
	val_accuracy_Pval = []
	val_accuracy_inf = []
	val_accuracy_LDpred = []
	val_accuracy_cojo = []
	val_accuracy_cojopred = []
	val_accuracy_cojopred2 = []
	val_accuracy_LDpredref = []
	val_accuracy_cojopred3 = []
	val_accuracy_cojopred3ref = []
	val_accuracy_cojoref = []
	val_accuracy_infref = []
	alpha_list = []

	# beta_hats = simulate_beta_hats(true_betas = train_betas, n = n, m = m, r2 = r2, LD_matrix = sample_D, m_ld_chunk_size = m_ld_chunk_size)

	beta_hats = betas_marg

	sample_D = sp.dot(traingeno, traingeno.T) / float(n)

	if verbose == True:
		print "calculating LDpred"
	ref_genotype = genotypes.simulate_genotypes_w_ld(n=10, m=m, n_samples=1, m_ld_chunk_size=m_ld_chunk_size, r2=r2)[0][0]
	ldDictref = get_LDpred_ld_tables(ref_genotype, ld_radius=m_ld_chunk_size, h2=h2, n_training=n)
	ldDict = get_LDpred_ld_tables(traingeno, ld_radius=m_ld_chunk_size, h2=h2, n_training=n)

	betainf = LDinf(beta_hats=beta_hats, ld_radius=m_ld_chunk_size, LD_matrix=sample_D, n=n, h2=h2)
	betainfref = LDinf(beta_hats=beta_hats, ld_radius=m_ld_chunk_size, LD_matrix=sample_Dref, n=n, h2=h2)

	betaLD = LDpred.ldpred_gibbs(beta_hats, start_betas=betainf, n=n, ld_radius=m_ld_chunk_size, p=p, ld_dict=ldDict["ld_dict"], h2=h2)
	betaLDref = LDpred.ldpred_gibbs(beta_hats, start_betas=betainfref, n=n, ld_radius=m_ld_chunk_size, p=p, ld_dict=ldDictref["ld_dict"], h2=h2)

	# estimate cojo corrected beta_hats
	if verbose == True:
		print "calculating cojo"
	cojo_beta_hats, cojo_betainf, n_cojo_selected_indices = ml_iter(beta_hats, traingeno, ld_radius=m_ld_chunk_size, h2=h2, p_threshold=p_threshold, printout = var_printout, LD = sample_D, ref_ld = False, LDmethod = False, r2 = r2)
	cojopred_beta_hats = cojo_beta_hats[:]

	cojo_beta_hats_LDpred_ref, cojo_betainf_LDpred_ref, n_cojo_selected_indices_ref = ml_iter(beta_hats, traingeno, ld_radius=m_ld_chunk_size, h2=h2, p_threshold=p_threshold, printout = var_printout, LD = sample_Dref, ref_ld = True, LDmethod = True, p = p, r2 = r2 )
	cojo_beta_hats_LDpred, cojo_betainf_LDpred, n_cojo_selected_indices = ml_iter(beta_hats, traingeno, ld_radius=m_ld_chunk_size, h2=h2, p_threshold=p_threshold, printout = var_printout, LD = sample_D, ref_ld = True, LDmethod = True, p = p, r2 = r2)
	if verbose == True:
		print "calculating pval thresholding"
	if p_threshold < 1:

		Z = n * (beta_hats ** 2)
		pval = []
		indices = []
		for val in Z:
			pval.append(stats.chi2.sf(val, 1))
		for j in range(len(pval)):
			if pval[j] < p_threshold:
				indices.append(j)

		beta_hatsPruned = beta_hats[indices]
	else:
		beta_hatsPruned = beta_hats
	# create j for use in the single marker test
	j = sp.argmax([abs(x) for x in beta_hats])

	if variance:
	# apply shrink of cojo betainf estimates

	# Cojo variance explained; yang et. al. 2012
		yTy = sp.dot(train_phen.T, train_phen)
		D = sp.dot(traingeno, traingeno.T)/float(n)

		Evarbb = sp.sum(cojo_beta_hats**2) - (1.0/n)


		bDbeta = sp.dot(sp.dot(sp.array(cojo_beta_hats).T, sp.identity(m)), betas_marg)
		R2J = bDbeta / yTy
		print "variance explained", Evarbb
		#sigma2 = max(h2 - R2J, h2*0.1)
		#print 'sigma2', sigma2
		sigma2 = max(h2 - Evarbb, h2*0.1) * 1.1
		sigma2 = min(max(Evarbb, h2 * 0.1), 1.0)
		#print 'sigma2 new', sigma2
		alpha = sigma2
		alpha = 1.0-(p**(1.0/2.0))
		print alpha
	if verbose == True:
		print "validating"
	for i in xrange(n_samples):

		# construct true validation phenotypes
		validation_phen_noise = stats.norm.rvs(0, sp.sqrt(1.0 - h2), size=2000)
		validation_phen_noise = sp.sqrt((1.0 - h2) / sp.var(validation_phen_noise)) * validation_phen_noise

		validation_genetic_part = sp.dot(geno[i].T, train_betas)
		validation_genetic_part = sp.sqrt(h2 / sp.var(validation_genetic_part)) * validation_genetic_part

		validation_phen = validation_genetic_part + validation_phen_noise

		YhatsSMT.append([])
		# for sample in the validation data. i.e. for each individual sample
		for k in range(len(validation_phen)):

			YhatsSMT[i].append(geno[i][j, k] * beta_hats[j])
		val_accuracy_SMT.append(stats.pearsonr(validation_phen, YhatsSMT[i])[0] ** 2)

		YhatsPval.append([])
		for k in range(len(validation_phen)):
			phen = 0
			for j in range(len(beta_hatsPruned)):
				phen += geno[i][j, k] * beta_hatsPruned[j]
			YhatsPval[i].append(phen)

		val_accuracy_Pval.append(stats.pearsonr(validation_phen, YhatsPval[i])[0] ** 2)

		# get PRS phenotypes
		YhatsPRS.append(sp.dot(geno[i].T, beta_hats))
		val_accuracy_PRS.append(stats.pearsonr(validation_phen, YhatsPRS[i])[0] ** 2)

		Yhatsinf.append(sp.dot(geno[i].T, betainf))
		val_accuracy_inf.append(stats.pearsonr(validation_phen, Yhatsinf[i])[0] ** 2)

		Yhatsinfref.append(sp.dot(geno[i].T, betainfref))
		val_accuracy_infref.append(stats.pearsonr(validation_phen, Yhatsinfref[i])[0] ** 2)

		YhatsLDpred.append(sp.dot(geno[i].T, betaLD["betas"]))
		val_accuracy_LDpred.append(stats.pearsonr(validation_phen, YhatsLDpred[i])[0] ** 2)

		YhatsLDpredref.append(sp.dot(geno[i].T, betaLDref["betas"]))
		val_accuracy_LDpredref.append(stats.pearsonr(validation_phen, YhatsLDpredref[i])[0] ** 2)

		Yhatscojo.append(sp.dot(geno[i].T, cojo_beta_hats))
		if sp.sum(Yhatscojo[i]) != 0:
			val_accuracy_cojo.append(stats.pearsonr(validation_phen, Yhatscojo[i])[0] ** 2)

		Yhatscojoref.append(sp.dot(geno[i].T, cojo_beta_hats_LDpred_ref))
		if sp.sum(Yhatscojo[i]) != 0:
			val_accuracy_cojoref.append(stats.pearsonr(validation_phen, Yhatscojoref[i])[0] ** 2)


		Yhatscojopred.append(sp.dot(geno[i].T, cojo_beta_hats))
		Yhatscojopred_betainf.append(sp.dot(geno[i].T, cojo_betainf))
		val_accuracy_cojopred2.append(stats.pearsonr(sp.dot(geno[i].T, cojo_beta_hats)*alpha + sp.dot(geno[i].T, betainf) * (1-alpha), validation_phen))

		Yhatscojopred3.append(sp.dot(geno[i].T, cojo_beta_hats_LDpred))
		Yhatscojopred3_betainf.append(sp.dot(geno[i].T, cojo_betainf_LDpred))

		Yhatscojopred3ref.append(sp.dot(geno[i].T, cojo_beta_hats_LDpred_ref))
		Yhatscojopred3ref_betainf.append(sp.dot(geno[i].T, cojo_betainf_LDpred_ref))
		"""
		Post normalization of Y hat values
		"""
		if n_cojo_selected_indices > 0:
			Yhatscojopred[i] = (Yhatscojopred[i] / sp.var(Yhatscojopred[i]))

			Yhatscojopred_betainf[i] = (Yhatscojopred_betainf[i] / sp.var(Yhatscojopred_betainf[i]))

			Yhatscojopred_concatenated.append((sp.array(Yhatscojopred[i]) * alpha) + (sp.array(Yhatscojopred_betainf[i]) * (1.0 - alpha)))
		else:
			Yhatscojopred_concatenated.append(sp.array(Yhatscojopred[i]) + sp.array(Yhatscojopred_betainf[i]))
		val_accuracy_cojopred.append(stats.pearsonr(validation_phen, Yhatscojopred_concatenated[i])[0] ** 2)
		if n_cojo_selected_indices == 0:
			assert Yhatscojopred_concatenated[i].any() == Yhatsinf[i].any()

		if n_cojo_selected_indices > 0:
			Yhatscojopred3[i] = (Yhatscojopred3[i] / sp.var(Yhatscojopred3[i]))

			Yhatscojopred3_betainf[i] = (Yhatscojopred3_betainf[i] / sp.var(Yhatscojopred3_betainf[i]))

			Yhatscojopred3_concatenated.append((sp.array(Yhatscojopred3[i]) * alpha) + (sp.array(Yhatscojopred3_betainf[i]) * (1.0 - alpha)))
		else:
			Yhatscojopred3_concatenated.append(sp.array(Yhatscojopred3[i]) + sp.array(Yhatscojopred3_betainf[i]))
		val_accuracy_cojopred3.append(stats.pearsonr(validation_phen, Yhatscojopred3_concatenated[i])[0] ** 2)

		if n_cojo_selected_indices > 0:
			Yhatscojopred3ref[i] = (Yhatscojopred3ref[i] / sp.var(Yhatscojopred3ref[i]))

			Yhatscojopred3ref_betainf[i] = (Yhatscojopred3ref_betainf[i] / sp.var(Yhatscojopred3ref_betainf[i]))

			Yhatscojopred3ref_concatenated.append((sp.array(Yhatscojopred3ref[i]) * alpha) + (sp.array(Yhatscojopred3ref_betainf[i]) * (1.0 - alpha)))
		else:
			Yhatscojopred3ref_concatenated.append(sp.array(Yhatscojopred3ref[i]) + sp.array(Yhatscojopred3ref_betainf[i]))
		val_accuracy_cojopred3ref.append(stats.pearsonr(validation_phen, Yhatscojopred3_concatenated[i])[0] ** 2)


	alpha_list.append(alpha)
	accSMT = sp.mean(val_accuracy_SMT)
	accPRS = sp.mean(val_accuracy_PRS)
	accPval = sp.mean(val_accuracy_Pval)
	accinf = sp.mean(val_accuracy_inf)
	accLDpred = sp.mean(val_accuracy_LDpred)
	accLDpredref = sp.mean(val_accuracy_LDpredref)
	acccojopred2 = sp.mean(val_accuracy_cojopred2)
	acccojopred3 = sp.mean(val_accuracy_cojopred3)
	acccojopred3ref = sp.mean(val_accuracy_cojopred3ref)
	acccojoref = sp.mean(val_accuracy_cojoref)
	accinfref = sp.mean(val_accuracy_infref)
	if len(val_accuracy_cojo) != 0:
		acccojo = sp.mean(val_accuracy_cojo)
	else:
		acccojo = None
	acccojopred = sp.mean(val_accuracy_cojopred)
	returndict = {'single marker test' : accSMT, 'polygenic risk scores' : accPRS, 'pvalue thresholding' : accPval,
	'LD-infinitesimal' : accinf, 'LD-pred' : accLDpred, 'cojo' : acccojo, 'cojopred' : acccojopred,
	 'cojopred2' : acccojopred2, 'alpha' : alpha, 'n selected SNPS in cojo' : n_cojo_selected_indices, 'cojopred3' : acccojopred3, 'cojopred3ref' : acccojopred3ref,
	 'LDpredref' : accLDpredref, 'cojoref' : acccojoref, 'LDinfref' : accinfref}
	print returndict
	return  accSMT, accPRS, accPval, accinf, accLDpred, acccojo, acccojopred, acccojopred2, acccojopred3, acccojopred3ref, accLDpredref, acccojoref, accinfref, alpha, n_cojo_selected_indices

def printtable(filename, p, N, M, Ntraits, validationN=5, variance = False, alpha = 0.5, ref_ld = True):
	with open(filename, 'w') as f:
		print >> f, 'N \t M \t P \t \t SMT \t PRS \t Pval \t LDinf \t LDpred \t cojo \t cojopred \t cojopred2 \t cojopred3 \t cojopred3ref \t LDpredref \t cojoref \t LDinfref \t alpha \n' 
		for i in range(len(N)):

			for m in range(len(M)):

				validation = genotypes.simulate_genotypes_w_ld(n=2000, m=M[m], n_samples=10, validation = True)
				for j in 																																																																																																																																																																																																																																												p:
					print j
					for l in range(Ntraits):
						print "trait", l, "N", N[i], "M", M[m], "p", j
						output = test_accuracy(N[i], M[m], n_samples=validationN, genotype=validation, h2=0.5, p=j,
						 r2=0.9, m_ld_chunk_size=100, p_threshold=5e-8, validation_set=None, alpha=0.5, verbose=False, variance = variance, ref_ld = ref_ld)
						print >> f, N[i], "\t", M[m], "\t", j, "\t", output[0], "\t", output[1], "\t", output[2], "\t", output[3], "\t", output[4], "\t", output[5], "\t", output[6], "\t", output[7], "\t", output[8], "\t", output[9], "\t", output[10], "\t", output[11], "\t", output[12], "\t", output[13], "\n"

def alpha_experiment(n, m, h2, p, r2, m_ld_chunk_size, p_threshold, filename, n_samples=10):

	with open(filename, 'w') as f:
		print >> f, 'N \t M \t P \t Alpha, accuracy \n'
		for j in p:
			print j
			training_set = genotypes.simulate_genotypes_w_ld(n=n, m=m, n_samples=1, m_ld_chunk_size=m_ld_chunk_size, r2=r2)
			traingeno = training_set[0][0]
			train_phen, train_betas = simulate_phenotypes(genotype=traingeno, n=n, m=m, h2=h2, p=j)

			betas_marg = (1.0 / n) * sp.dot(train_phen, traingeno.T)
			output = estimate_alpha(traingeno, train_phen, betas_marg, train_betas, n, m, h2, j, r2, m_ld_chunk_size, p_threshold, testing=True, n_samples=n_samples)
			for i in range(len(output[1])):
				print >> f, n, "\t", m, "\t", j, "\t", output[1][i], "\t", output[0][i], "\n"

def r2_experiment(n, m, h2, p, m_ld_chunk_size, p_threshold, filename):
	with open(filename, 'w') as f:
		print >> f, 'N \t M \t P \t r2 \t SMT \t PRS \t Pval \t LDinf \t LDpred \t cojo \t cojopred \t cojopred2 \t cojopred3 \t cojopred3ref \t LDpredref \t cojoref \t LDinfref \t alpha\n'
		r2 = [0.1 * x for x in range(10)]
		print r2
		for j in p:
			print j
			for i in r2:
				validation = genotypes.simulate_genotypes_w_ld(n=2000, m=m, r2=i, n_samples=10, validation = True)
				for k in range(5):
					output = test_accuracy(n=n, m=m, n_samples=10, genotype=validation, h2=0.5, p=j,
					 r2=i, m_ld_chunk_size=100, p_threshold=5e-8, alpha=0.5, verbose=False, variance = True)
					print >> f, n, "\t", m, "\t", j, "\t", i, "\t", output[0], "\t", output[1], "\t", output[2], "\t", output[3], "\t", output[4], "\t", output[5], "\t", output[6], "\t", output[7],  "\t", output[8], "\t", output[9], "\t", output[10], "\t", output[11], "\t", output[12], "\t", output[13], "\t", output[14], "\n"

def estimate_alpha(train_set, train_phen, betas_marg, train_betas, n, m, h2, p, r2, m_ld_chunk_size, p_threshold, testing=False, n_samples=1):
	# Try a smaller set of alphas, e.g. {0.1,0.3,0.5,0.7,0.9}
	alpha_list = [x * 0.05 for x in range(1, 21)]
	beta_hats = betas_marg
	sample_D = sp.dot(train_set, train_set.T) / float(n)
	cojo_beta_hats, cojo_betainf, n_cojo_selected_indices = ml_iter(betas_marg, train_set, ld_radius=m_ld_chunk_size, h2=h2, p_threshold=p_threshold, printout = False, LD = sample_D)
	betas = train_betas
	cojopred_beta_hats = cojo_beta_hats[:]
	n_samples = 1
	accuracy = []

	for alpha in alpha_list:
		print alpha
		CVaccuracy = []
		#sample = genotypes.simulate_genotypes_w_ld(n=n, m=m, n_samples=n_samples, m_ld_chunk_size=m_ld_chunk_size, r2=r2)
		for i in range(n_samples):
			#geno = sample[0][i]
			geno = train_set
			Yhatscojopred = sp.dot(geno.T, cojo_beta_hats)
			Yhatscojopred_betainf = (sp.dot(geno.T, cojo_betainf))


			"""
			Post normalization of Y hat values
			"""
			if n_cojo_selected_indices == 0:
				Yhatscojopred = sp.array([0])
			else:
				Yhatscojopred = (Yhatscojopred / sp.var(Yhatscojopred)) / max(1, n_cojo_selected_indices)

			Yhatscojopred_betainf = (Yhatscojopred_betainf / sp.var(Yhatscojopred_betainf)) / (m - n_cojo_selected_indices)
			Yhatscojopred_concatenated = (Yhatscojopred * alpha) + (Yhatscojopred_betainf * (1 - alpha))





			#validation_phen_noise = stats.norm.rvs(0, sp.sqrt(1.0 - h2), size=n)
			#validation_phen_noise = sp.sqrt((1.0 - h2) / sp.var(validation_phen_noise)) * validation_phen_noise

			#validation_genetic_part = sp.dot(geno.T, betas)
			#validation_genetic_part = sp.sqrt(h2 / sp.var(validation_genetic_part)) * validation_genetic_part

			#validation_phen = validation_genetic_part + validation_phen_noise
			validation_phen = train_phen

			print Yhatscojopred.shape
			#CVaccuracy.append(stats.pearsonr(validation_phen, Yhatscojopred_concatenated)[0] ** 2)
			accuracy.append(stats.pearsonr(validation_phen, Yhatscojopred_concatenated)[0] ** 2)
		#accuracy.append(sp.mean(CVaccuracy))
	best_accuracy_index = accuracy.index(max(accuracy))
	best_alpha = alpha_list[best_accuracy_index]
	if testing == True:
		return accuracy, alpha_list
	print "acc list", accuracy
	print "best alpha : ", best_alpha
	print "accuracy : ", max(accuracy)
	return best_alpha

def beta_experiment(n = 4000, m = 8000, p = [0.0001, 0.5, 1.0], normalized = False):
	"""
	sim a phenotype and inspect how the different beta estimates fluctuate around each other
	"""
	true_betas = []
	marginal_betas = []
	pval_betas = []
	LDpred_betas = []
	LDinf_betas = []
	cojo_betas = []
	cojopred_betas = []
	cojo_betas_ref = []
	LDpred_ref = []
	for i in p:
		print "start"
		training_set = genotypes.simulate_genotypes_w_ld(n=n, m=m, n_samples=1, m_ld_chunk_size= 100, r2=0.9)
		traingeno = training_set[0][0]
		sample_D = training_set[1][0]
		train_phen, train_betas = simulate_phenotypes(genotype=traingeno, n=n, m=m, h2=0.5, p= i)
		betas_marg = (1.0 / n) * sp.dot(train_phen, traingeno.T)
		beta_hats = betas_marg
		refgenome = genotypes.simulate_genotypes_w_ld(n=5, m=m, n_samples=1, m_ld_chunk_size= 100, r2=0.9)[0][0]
		sample_D = sp.dot(refgenome, refgenome.T) / float(n)
		ldDict = get_LDpred_ld_tables(refgenome, ld_radius= 100, h2= 0.5, n_training= n)
		print "betainf"
		betainf = LDinf(beta_hats=beta_hats, ld_radius=100, LD_matrix=sample_D, n=n, h2=0.5)
		print "LDpred"
		betaLD_ref = LDpred.ldpred_gibbs(beta_hats, start_betas=betainf, n=n, ld_radius= 100, p= i, ld_dict=ldDict["ld_dict"], h2=0.5)
		print "cojo"
		cojo_beta_hats_ref, cojo_betainf, n_cojo_selected_indices = ml_iter(beta_hats, traingeno, ld_radius= 100, h2=0.5, p_threshold=5e-8, printout = False, LD = sample_D)


		sample_D = sp.dot(traingeno, traingeno.T) / float(n)
		print "betainf"
		betainf = LDinf(beta_hats=beta_hats, ld_radius=100, LD_matrix=sample_D, n=n, h2=0.5)
		ldDict = get_LDpred_ld_tables(traingeno, ld_radius= 100, h2= 0.5, n_training= n)
		print "LDpred"
		betaLD = LDpred.ldpred_gibbs(beta_hats, start_betas=betainf, n=n, ld_radius= 100, p= i, ld_dict=ldDict["ld_dict"], h2=0.5)

		cojo_beta_hats, cojo_betainf, n_cojo_selected_indices = ml_iter(beta_hats, traingeno, ld_radius= 100, h2=0.5, p_threshold=5e-8, printout = False, LD = sample_D)
		print "done"
		Z = n * (beta_hats ** 2)
		pval = []
		indices = []
		beta_hatsPruned = sp.zeros(m)
		for val in Z:
			pval.append(stats.chi2.sf(val, 1))
		for j in range(len(pval)):
			if pval[j] < 5e-8:
				indices.append(j)
		beta_hatsPruned[indices] = beta_hats[indices]

		if normalized:
			betas_marg = (betas_marg - train_betas)**2
			beta_hatsPruned = (beta_hatsPruned - train_betas)**2
			betaLD['betas'] = (betaLD['betas'] - train_betas)**2
			betainf = (betainf - train_betas)**2
			cojo_beta_hats = (cojo_beta_hats - train_betas)**2
			cojo_betainf = (cojo_betainf - train_betas)**2
			cojo_beta_hats_ref = (cojo_beta_hats_ref - train_betas)**2
			betaLD_ref['betas'] = (betaLD_ref['betas'] - train_betas)**2
			train_betas = train_betas - train_betas

			assert train_betas.all() == 0
		true_betas.append(train_betas)
		marginal_betas.append(betas_marg)
		pval_betas.append(beta_hatsPruned)
		LDpred_betas.append(betaLD['betas'])
		LDinf_betas.append(betainf)
		cojo_betas.append(cojo_beta_hats)
		cojopred_betas.append(cojo_betainf)
		cojo_betas_ref.append(cojo_beta_hats_ref)
		LDpred_ref.append(betaLD_ref['betas'])
	return true_betas, marginal_betas, pval_betas, LDpred_betas, LDinf_betas, cojo_betas, cojopred_betas, LDpred_ref, cojo_betas_ref

if __name__ == "__main__":
	""
	#N = [1000]
	#M = [2000]
	#p = [0.0005, 0.001, 0.01, 0.05, 0.1, 0.5, 0.7, 1.0]
	#alpha_experiment(n= 2000, m= 4000, h2=0.5, p=p, r2=0.9, m_ld_chunk_size=100, p_threshold=5e-8, filename="alpha_as_a_function_of_p", n_samples=50)
	#p = [0.0005, 0.5, 1.0]
	#r2_experiment(n = 1000, m = 2000, h2 = 0.5, p = p, m_ld_chunk_size = 100, p_threshold = 5e-8, filename = "r2_and_acc.csv")
	#printtable("effectofP_NM05_new_method_4_variance.csv", p = p, N = N, M = M, Ntraits = 10, variance = True)
	#N = [200]
	#printtable("effectofP_NM01_new_method_4_variance.csv", p = p, N = N, M = M, Ntraits = 10, variance = True)
	#N = [1600]
	#printtable("effectofP_NM08_new_method_4_variance.csv", p = p, N = N, M = M, Ntraits = 10, variance = True)
	#N = [3000]
	#printtable("effectofP_NM15_new_method_4_variance.csv", p = p, N = N, M = M, Ntraits = 10, variance = True)
	
