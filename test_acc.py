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
		lds_i = sp.sum(r2s - (1-r2s) / (n-2),dtype='float32')
		#lds_i = sp.sum(r2s - (1-r2s)*empirical_null_r2)
		ld_scores[snp_i] =lds_i
	ret_dict = {'ld_dict':ld_dict, 'ld_scores':ld_scores}

	if ld_window_size>0:
		ref_ld_matrices = []
		inf_shrink_matrices = []
		for i, wi in enumerate(range(0, m, ld_window_size)):
			start_i = wi
			stop_i = min(m, wi + ld_window_size)
			curr_window_size = stop_i - start_i
			X = snps[start_i: stop_i]
			D = sp.dot(X, X.T) / n
			ref_ld_matrices.append(D)
			if h2!=None and n_training!=None:
				A = ((m / h2) * sp.eye(curr_window_size) + (n_training / (1)) * D)
				A_inv = linalg.pinv(A)
				inf_shrink_matrices.append(A_inv)
		ret_dict['ref_ld_matrices']=ref_ld_matrices
		if h2!=None and n_training!=None:
			ret_dict['inf_shrink_matrices']=inf_shrink_matrices
	return ret_dict


def simulate_traits_fast(n, m, h2 = 0.5, p = 1.0):
	#simulate betas from a normal distribution with mean 0 and variance = h2/m.
	#p chooses architecture where 1 is infinitisimal, 0<p<1 is non-infinitisimal and p = 1/m is mendelian
	if p == 1.0:
		M = m
		betas = stats.norm.rvs(0, sp.sqrt(h2/m), size = m)
	else:
		M = int(round(m*p))
		if M == 0:
			M = 1
		betas = sp.concatenate((stats.norm.rvs(0, sp.sqrt(h2 / M), size=M), sp.zeros(m - M, dtype=float)))
	#coefficients are scaled to have mean 0 and variance 1

	betas_var = sp.var(betas)
	betas_scalar = sp.sqrt(h2/ (m * betas_var))
	betas = betas * betas_scalar
	#betas_list.append(betas)

	return betas

def LDinf(beta_hats, ld_radius, LD_matrix, n, h2):
	m = len(beta_hats)
	betainf = sp.zeros(m)

	for m_i in range(0,m, ld_radius):
		#calculate the ld corrected betaHats under the infinitesimal model
		
		m_end = min(m_i + ld_radius, m)
		if LD_matrix.shape == (ld_radius, ld_radius):
			A = ((m/h2) * sp.eye(min(ld_radius, m_end-m_i)) + (n/(1)) * LD_matrix)
		else:
			A = ((m/h2) * sp.eye(min(ld_radius, m_end-m_i)) + (n/(1)) * LD_matrix[m_i:m_end, m_i:m_end])
		try:
			Ainv = linalg.pinv(A)
		except Exception(err_str):
			print err_str
			print linalg.pinv(LD_matrix)
		betainf[m_i:m_end] = sp.dot(Ainv * n, beta_hats[m_i:m_end])
	return betainf

def ml_iter(beta_hats, snps, ld_radius, h2, p_threshold):

	non_selected_indices = []
	snps = sp.array(snps)

	#Ordering the beta_hats, and storing the order
	m = len(beta_hats)
	beta_hats = beta_hats.tolist()
	ld_table = {}
	for i in range(m):
		ld_table[i] = {'ld_partners':[i], 'beta_hat_list':[beta_hats[i]], 'sumsq' : beta_hats[i]**2, 'pval' : 0, 'D' : sp.array([1.0]), 'D_inv':sp.array([1.0])}
		ld_table[i]['pval'] = stats.chi2.sf((ld_table[i]['beta_hat_list'][0]**2)*snps.shape[0], 1)
	n_test = len(snps[0])

	selected_indices = set()
	updated_beta_hats = beta_hats[:]

	stop_boolean = True
	while stop_boolean == True:

		#Sort and select beta
		l = zip((sp.array(updated_beta_hats) ** 2).tolist(), range(m))
		l.sort(reverse=True)
		for beta_hat, beta_i in l:
			if not beta_i in selected_indices:
				if ld_table[beta_i]['pval'] <= p_threshold:
					selected_indices.add(beta_i)
					break

				if ld_table[i]['pval'] > p_threshold:
					stop_boolean = False
					break

		if stop_boolean == False:
			break

		#Iterating over the window around the selected beta
		start_i = max(0, beta_i - ld_radius)
		end_i = min(beta_i + ld_radius, m)
		for i in range(start_i, end_i):
			if i == beta_i:
				continue
			else:
				#Update the LD matrix
				d = ld_table[i]
				ld_partners = d['ld_partners']
				beta_hat_list = d['beta_hat_list']
				ld_partners.append(beta_i)
				beta_hat_list.append(beta_hats[beta_i])
				bs = sp.array(beta_hat_list)
				X = snps[ld_partners]
				D = sp.dot(X, X.T) / float(n_test)
				ld_table['D'] = D
				D_inv = linalg.pinv(D)
				ld_table['D_inv'] = D_inv
				
				updated_beta = sp.dot(D_inv, bs)
				sumsq2 = sp.sum(updated_beta**2)
				added_var_explained = sumsq2 - d['sumsq']
				d['pval'] = stats.chi2.sf(added_var_explained, 1)
				d['sumsq'] = sumsq2
				updated_beta_hats[i] = updated_beta[0]

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
				Xi= snps[i]
				Xpartners = snps[ld_partners]
				upb = sp.array(updated_beta_hats)[ld_partners]
				Di = sp.dot(Xi, Xpartners.T) / float(n_test)
				updated_beta_hats[i] = beta_hats[i] - sp.sum(sp.dot(Di, upb))


	#Remove cojosnps
	snps = sp.array(snps)
	newX = snps[non_selected_indices]
	assert newX.shape[0] + len(selected_indices) == len(beta_hats)
	newbetas = sp.array(updated_beta_hats)[non_selected_indices]

	m,n = newX.shape
	D = sp.dot(newX, newX.T) / float(n)

	betainf = LDinf(beta_hats = newbetas, ld_radius = ld_radius, LD_matrix = D, n = n, h2 = h2)


	"""
	normalizing beta hat values
	Removed and instead attempt to normalize predicted Y hats.
	"""
	# #LDpred standard deviation
	# betainf_std = sp.sqrt(sp.var(betainf))

	# #normalize betainf weights
	# betainf = (betainf / betainf_std) / m
	# cojo_beta_hats = sp.array(updated_beta_hats)[selected_indices]




	# #normalize cojo weights
	# if len(cojo_beta_hats) < 2:
	#   cojo_std = 1  
	# else:
	#	 print cojo_beta_hats
	#	 cojo_std = sp.sqrt(sp.var(cojo_beta_hats))
	# cojo_beta_hats_norm = (cojo_beta_hats / cojo_std) / len(selected_indices)
	# print "cojo snps", len(selected_indices)
	# print "difference", sp.absolute(sp.sum(cojo_beta_hats_norm**2) - sp.sum(betainf**2))
	
	# assert sp.absolute(sp.sum(cojo_beta_hats_norm**2) - sp.sum(betainf**2)) < 0.05

	# assert len(selected_indices) + len(non_selected_indices) == len(beta_hats), "error"

	# updated_beta_hats_conc = sp.zeros(len(beta_hats))
	# for i in range(len(beta_hats)):
	#	 if i in selected_indices:
	#		 index = selected_indices.index(i)
	#		 updated_beta_hats_conc[i] = cojo_beta_hats_norm[index]
	#	 else:
	#		 index = non_selected_indices.index(i)
	#		 updated_beta_hats_conc[i] = betainf[index]
	
	betainf_with_0 = sp.zeros(len(beta_hats))
	updated_beta_hats_with_0 = sp.zeros(len(beta_hats))
	betainf_with_0[non_selected_indices] = betainf

	updated_beta_hats_with_0 = cojo_updated_beta_hats
	cojo_updated_beta_hats = sp.array(cojo_updated_beta_hats)

	return cojo_updated_beta_hats, betainf_with_0, len(selected_indices)

def simulate_phenotypes(genotype, n, m, h2, p):
	betas = simulate_traits_fast(n, m, h2, p)
	phen_noise = stats.norm.rvs(0, sp.sqrt(1.0 - h2), size=n) 
	phen_noise = sp.sqrt((1.0 - h2) / sp.var(phen_noise)) * phen_noise
	genetic_part = sp.dot(genotype.T, betas)
	genetic_part = sp.sqrt(h2 / sp.var(genetic_part)) * genetic_part
	train_phen = genetic_part + phen_noise
	train_phen = (1/sp.std(train_phen)) * train_phen
	return train_phen, betas

def simulate_beta_hats(true_betas, n, m, r2, LD_matrix, m_ld_chunk_size):
	noises = stats.norm.rvs(0,1,size=m)
	if r2 == 0:
		beta_hats = betas + sp.sqrt(1.0/n) * noises
	else:
		#if ld
		C = sp.sqrt(((1.0)/n))*linalg.cholesky(LD_matrix)
		D_I = linalg.pinv(LD_matrix)
		betas_ld = sp.zeros(m)
		noises_ld = sp.zeros(m)
		for m_i in range(0,m,m_ld_chunk_size):
			m_end = m_i+m_ld_chunk_size
			betas_ld[m_i:m_end] = sp.dot(LD_matrix,true_betas[m_i:m_end])
			noises_ld[m_i:m_end]  = sp.dot(C.T,noises[m_i:m_end])
		beta_hats = betas_ld + noises_ld
	return beta_hats



def test_accuracy(n, m, n_samples = 10,  genotype = None,  h2 = 0.5, p = 1.0, r2 = 0.9, m_ld_chunk_size = 100, p_threshold = 5e-8, validation_set = None, alpha = None, verbose = False):
	#Simulate training data
	if verbose == True:
		print "simulating training set"
	training_set = genotypes.simulate_genotypes_w_ld(n = n, m = m, n_samples = 1, m_ld_chunk_size = m_ld_chunk_size, r2 = r2)
	traingeno = training_set[0][0]
	sample_D = training_set[1][0]
	train_phen, train_betas = simulate_phenotypes(genotype = traingeno, n = n, m = m, h2 = h2, p = p)

	#estimate marginal beta hats
	betas_marg = (1.0 / n) * sp.dot(train_phen, traingeno.T)

	#Estimate or import alpha hyperparameter
	if verbose == True:
		print "estimating alpha"
	if alpha == None:
		alpha = estimate_alpha(n = n, m = m,  h2 = h2, p = p, r2 = r2, m_ld_chunk_size = m_ld_chunk_size, p_threshold = p_threshold)

	#simulate validation data or import data
	if genotype == None:
		if r2 == 0:
			geno = genotypes.simulate_genotypes(n = 2000, m = m, n_samples = n_samples)
		else:
			sample = genotypes.simulate_genotypes_w_ld(n = 2000, m = m, n_samples = n_samples, m_ld_chunk_size = m_ld_chunk_size, r2 = r2)
			geno = sample[0]
	else:
		if r2 == 0:
			geno = genotype[0]
		else:
			geno = genotype[0]

	#phenotype lists
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

	#validation accuracy list
	val_accuracy_SMT = []
	val_accuracy_PRS = []
	val_accuracy_Pval = []
	val_accuracy_inf = []
	val_accuracy_LDpred = []
	val_accuracy_cojo = []
	val_accuracy_cojopred = []



	#beta_hats = simulate_beta_hats(true_betas = train_betas, n = n, m = m, r2 = r2, LD_matrix = sample_D, m_ld_chunk_size = m_ld_chunk_size)

	beta_hats = betas_marg
	sample_D = sp.dot(traingeno, traingeno.T) / float(n)

	if verbose == True:
		print "calculating LDpred"
	betainf = LDinf(beta_hats = beta_hats, ld_radius = m_ld_chunk_size, LD_matrix = sample_D, n = n, h2 = h2)
	ldDict = get_LDpred_ld_tables(traingeno, ld_radius=m_ld_chunk_size, h2=h2, n_training=n)
	betaLD = LDpred.ldpred_gibbs(beta_hats, start_betas = betainf, n = n, ld_radius = m_ld_chunk_size, p = p, ld_dict = ldDict["ld_dict"], h2 = h2)
		

	#estimate cojo corrected beta_hats
	if verbose == True:
		print "calculating cojo"
	cojo_beta_hats, cojo_betainf, n_cojo_selected_indices = ml_iter(beta_hats, traingeno, ld_radius= m_ld_chunk_size, h2 = h2, p_threshold= p_threshold)
	cojopred_beta_hats = cojo_beta_hats[:]


	if verbose == True:
		print "calculating pval thresholding"
	if p_threshold < 1:

		Z = n*(beta_hats**2)
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
	#create j for use in the single marker test
	j = sp.argmax([abs(x) for x in beta_hats])

	#apply shrink of cojo betainf estimates

	#Cojo variance explained; yang et. al. 2012
	#yTy = sp.dot(train_phen.T, train_phen)
	#D = sp.dot(traingeno, traingeno.T)/float(n)

	#Alt variance explained
	#Evarbb = sp.sum(cojo_beta_hats**2) - (10.0/n)

	#print "Evar", Evarbb

	#bDbeta = sp.dot(sp.dot(sp.array(cojo_beta_hats).T, sp.identity(m)), betas_marg)
	#print 'bDbeta', bDbeta
	#print 'yTy', yTy
	#R2J = bDbeta / yTy
	#print R2J
	#sigma2 = max(h2 - R2J, h2*0.1)
	#print 'sigma2', sigma2
	#sigma2 = max(h2 - Evarbb, h2*0.1) * 1.1
	#print 'sigma2 new', sigma2
	#cojo_betainf = sp.array(cojo_betainf) * sigma2

	if verbose == True:
		print "validating"
	for i in xrange(n_samples):

		#construct true validation phenotypes
		validation_phen_noise =  stats.norm.rvs(0, sp.sqrt(1.0 - h2), size= 2000)
		validation_phen_noise = sp.sqrt((1.0 - h2) / sp.var(validation_phen_noise)) * validation_phen_noise

		validation_genetic_part = sp.dot(geno[i].T, train_betas)
		validation_genetic_part = sp.sqrt(h2 / sp.var(validation_genetic_part)) * validation_genetic_part

		validation_phen = validation_genetic_part + validation_phen_noise

		YhatsSMT.append([])
		#for sample in the validation data. i.e. for each individual sample
		for k in range(len(validation_phen)):

			YhatsSMT[i].append(geno[i][j,k] * beta_hats[j])
		val_accuracy_SMT.append(stats.pearsonr(validation_phen,YhatsSMT[i])[0]**2)

		YhatsPval.append([])
		for k in range(len(validation_phen)):
			phen = 0
			for j in range(len(beta_hatsPruned)):
				phen += geno[i][j,k] * beta_hatsPruned[j]
			YhatsPval[i].append(phen)

		val_accuracy_Pval.append(stats.pearsonr(validation_phen,YhatsPval[i])[0]**2)

		#get PRS phenotypes
		YhatsPRS.append(sp.dot(geno[i].T, beta_hats))
		val_accuracy_PRS.append(stats.pearsonr(validation_phen, YhatsPRS[i])[0]**2)

		Yhatsinf.append(sp.dot(geno[i].T, betainf))
		val_accuracy_inf.append(stats.pearsonr(validation_phen, Yhatsinf[i])[0]**2)

		YhatsLDpred.append(sp.dot(geno[i].T, betaLD["betas"]))
		val_accuracy_LDpred.append(stats.pearsonr(validation_phen, YhatsLDpred[i])[0]**2)

		Yhatscojo.append(sp.dot(geno[i].T, cojo_beta_hats))
		if sp.sum(Yhatscojo[i]) != 0:
			val_accuracy_cojo.append(stats.pearsonr(validation_phen, Yhatscojo[i])[0]**2)

		Yhatscojopred.append(sp.dot(geno[i].T, cojo_beta_hats))
		Yhatscojopred_betainf.append(sp.dot(geno[i].T, cojo_betainf))

		"""
		Post normalization of Y hat values
		"""
		if n_cojo_selected_indices > 0:
			Yhatscojopred[i] = (Yhatscojopred[i] / sp.var(Yhatscojopred[i]))

			Yhatscojopred_betainf[i] = (Yhatscojopred_betainf[i] / sp.var(Yhatscojopred_betainf[i]))

			Yhatscojopred_concatenated.append((sp.array(Yhatscojopred[i])*alpha) + (sp.array(Yhatscojopred_betainf[i]) * (1.0-alpha)))
		else:
			Yhatscojopred_concatenated.append(sp.array(Yhatscojopred[i]) + sp.array(Yhatscojopred_betainf[i]))
		val_accuracy_cojopred.append(stats.pearsonr(validation_phen, Yhatscojopred_concatenated[i])[0]**2)
		if n_cojo_selected_indices == 0:
			assert Yhatscojopred_concatenated[i].any() == Yhatsinf[i].any()

	accSMT = sp.mean(val_accuracy_SMT)
	accPRS = sp.mean(val_accuracy_PRS)
	accPval = sp.mean(val_accuracy_Pval)
	accinf = sp.mean(val_accuracy_inf)
	accLDpred = sp.mean(val_accuracy_LDpred)

	if len(val_accuracy_cojo) != 0:
		acccojo = sp.mean(val_accuracy_cojo)
	else:
		acccojo = None
	acccojopred = sp.mean(val_accuracy_cojopred)
	return  accSMT, accPRS, accPval, accinf, accLDpred, acccojo, acccojopred, n_cojo_selected_indices

def printtable(filename, p, N, M, Ntraits, validationN = 5):
	with open(filename, 'w') as f:
		print >>f, 'N \t M \t P \t \t SMT \t PRS \t Pval \t LDinf \t LDpred \t cojo \t cojopred \n' 
		for i in range(len(N)):
			print "N"
			print N[i]
			for m in range(len(M)):
				print "M"
				print M[m]
				validation = genotypes.simulate_genotypes_w_ld(n = 2000, m = M[m], n_samples = 10)
				for j in p:
					print j
					for l in range(Ntraits):
						output = test_accuracy(N[i], M[m], n_samples = validationN,  genotype = validation,  h2 = 0.5, p = j, r2 = 0.9, m_ld_chunk_size = 100, p_threshold = 5e-8, validation_set = None, alpha = None, verbose = False)
						print >>f, N[i],"\t",M[m],"\t",j,"\t",output[0],"\t",output[1], "\t", output[2], "\t", output[3], "\t", output[4], "\t", output[5], "\t", output[6], "\n"

def alpha_experiment(n, m, h2, p, r2, m_ld_chunk_size, p_threshold, filename):
	with open(filename, 'w') as f:
		print >>f, 'N \t M \t P \t Alpha, accuracy \n'
		for j in p:
			print j
			output = estimate_alpha(n, m, h2, j, r2, m_ld_chunk_size, p_threshold, testing = True)
			for i in range(len(output[1])):
				print >>f, n, "\t", m, "\t", j, "\t", output[1][i], "\t", output[0][i], "\n"

def estimate_alpha(n, m,  h2, p, r2, m_ld_chunk_size, p_threshold, testing = False):
	alpha_list = [x*0.05 for x in range(1, 21)]
	accuracy = []
	for alpha in alpha_list:
		print alpha
		CVaccuracy = []
		for i in range(15):
			betas = simulate_traits_fast(n, m, h2, p)

			training_set = genotypes.simulate_genotypes_w_ld(n = n, m = m, n_samples = 1, m_ld_chunk_size = m_ld_chunk_size, r2 = r2)
			traingeno = training_set[0][0]
			sample_D = training_set[1][0]
			#estimate marginal beta hats
			phen_noise = stats.norm.rvs(0, sp.sqrt(1.0 - h2), size=n) 
			phen_noise = sp.sqrt((1.0 - h2) / sp.var(phen_noise)) * phen_noise
			genetic_part = sp.dot(traingeno.T, betas)
			genetic_part = sp.sqrt(h2 / sp.var(genetic_part)) * genetic_part
			train_phen = genetic_part + phen_noise
			train_phen = (1/sp.std(train_phen)) * train_phen	
			betas_marg = (1.0 / n) * sp.dot(train_phen, traingeno.T)

			beta_hats = betas_marg
			sample_D = sp.dot(traingeno, traingeno.T) / float(n)

			cojo_beta_hats, cojo_betainf, n_cojo_selected_indices = ml_iter(betas_marg, traingeno, ld_radius= m_ld_chunk_size, h2 = h2, p_threshold= p_threshold)

			cojopred_beta_hats = cojo_beta_hats[:]


			sample = genotypes.simulate_genotypes_w_ld(n = n, m = m, n_samples = 1, m_ld_chunk_size = m_ld_chunk_size, r2 = r2)
			geno = sample[0][0]

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
			Yhatscojopred_concatenated = (Yhatscojopred*alpha) + (Yhatscojopred_betainf * (1-alpha))





			validation_phen_noise =  stats.norm.rvs(0, sp.sqrt(1.0 - h2), size= n)
			validation_phen_noise = sp.sqrt((1.0 - h2) / sp.var(validation_phen_noise)) * validation_phen_noise

			validation_genetic_part = sp.dot(geno.T, betas)
			validation_genetic_part = sp.sqrt(h2 / sp.var(validation_genetic_part)) * validation_genetic_part

			validation_phen = validation_genetic_part + validation_phen_noise



			CVaccuracy.append(stats.pearsonr(validation_phen, Yhatscojopred_concatenated)[0]**2)
		accuracy.append(sp.mean(CVaccuracy))
	best_accuracy_index = accuracy.index(max(accuracy))
	best_alpha = alpha_list[best_accuracy_index]
	if testing == True:
		return accuracy, alpha_list
	print "acc list", accuracy
	print "best alpha : ", best_alpha
	print "accuracy : ", max(accuracy)
	return best_alpha



if __name__ == "__main__":
	""
	#test = []
	#for i in range(50):
	#	test.append(test_accuracy(n = 1000, m = 2000, n_samples = 5,  genotype = None,  h2 = 0.5, p = 0.03, r2 = 0.9, m_ld_chunk_size = 100, p_threshold = 5e-8, validation_set = None, alpha = 0.5, verbose = True))

	#t = map(list,zip(*test))
	p = [x*0.0002 for x in range(1,11)]
	p = p + [x*0.001 for x in range(2,11)]
	p = p + [x*0.01 for x in range(2,11)]
	p = p + [x*0.2 for x in range(1,6)]
	N = [1000]
	M = [2000]
	alpha_experiment(n = 1000, m = 2000, h2 = 0.5, p = p, r2 = 0.9, m_ld_chunk_size = 100, p_threshold = 5e-8, filename = "alpha_as_a_function_of_p")
	# printtable("effectofP_NM0.1", p = p, N = N, M = M, Ntraits = 10)
	# print "0.1"
	# N = [3000]
	# printtable("effectofP_NM0.5", p = p, N = N, M = M, Ntraits = 10)
	# print "0.5"
	# N = [4800]
	# printtable("effectofP_NM0.8", p = p, N = N, M = M, Ntraits = 10)
	# print "0.8"
	#N = [9000]
	#printtable("effectofP_NM0.1.5", p = p, N = N, M = M, Ntraits = 10)
	#print "1.5"
	
