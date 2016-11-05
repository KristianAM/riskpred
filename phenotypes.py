import scipy as sp
from scipy import stats
import genotypes
from scipy import linalg
import LDpred
import cojo

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
	#if p == 1.0/m:
	#	M = m
	#	betas = sp.concatenate((stats.norm.rvs(0, sp.sqrt(h2 / 1), size=1), sp.zeros(m - 1, dtype=float)))
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

def estimate_alpha(n = 500, m = None,  h2 = 0.5, p = 1.0, r2 = 0.9, m_ld_chunk_size = 100, p_threshold = 0.05):
	alpha_list = [x*0.05 for x in range(1, 21)]
	accuracy = []
	for alpha in alpha_list:
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
		betas_marg = (1.0 / n) * sp.dot(train_phen, traingeno.T)
		noises = stats.norm.rvs(0,1,size=m)
		betainf = sp.zeros(m)
		if r2 == 0:
			beta_hats = betas + sp.sqrt(1.0/n) * noises
		else:
			#if ld
			C = sp.sqrt(((1.0)/n))*linalg.cholesky(sample_D)
			D_I = linalg.pinv(sample_D)
			betas_ld = sp.zeros(m)
			noises_ld = sp.zeros(m)
			for m_i in range(0,m,m_ld_chunk_size):
				m_end = m_i+m_ld_chunk_size
				betas_ld[m_i:m_end] = sp.dot(sample_D,betas[m_i:m_end])
				noises_ld[m_i:m_end]  = sp.dot(C.T,noises[m_i:m_end])
			beta_hats = betas_ld + noises_ld
			for m_i in range(0,m, m_ld_chunk_size):

				#calculate the ld corrected betaHats under the infinitesimal model
				m_end = m_i + m_ld_chunk_size
				A = ((m/h2) * sp.eye(m_ld_chunk_size) + (n/(1)) * sample_D)
				Ainv = linalg.pinv(A)
				betainf[m_i:m_end] = sp.dot(Ainv * n, beta_hats[m_i:m_end])

			ldDict = get_LDpred_ld_tables(traingeno, ld_radius=m_ld_chunk_size, h2=h2, n_training=n)
			betaLD = LDpred.ldpred_gibbs(beta_hats, start_betas = betainf, n = n, ld_radius = m_ld_chunk_size, p = p, ld_dict = ldDict["ld_dict"], h2 = h2)
		cojo_beta_hats, cojopred_beta_hats, cojo_betainf, n_cojo_selected_indices = cojo.ml_iter(betas_marg, traingeno, ld_radius= m_ld_chunk_size, h2 = h2, p_threshold= p_threshold)

		cojopred_beta_hats = cojo_beta_hats[:]


		sample = genotypes.simulate_genotypes_w_ld(n = n, m = m, n_samples = 1, m_ld_chunk_size = m_ld_chunk_size, r2 = r2)
		geno = sample[0][0]

		Yhatscojopred = sp.dot(geno.T, cojo_beta_hats)
		Yhatscojopred_betainf = (sp.dot(geno.T, cojo_betainf))
		"""
		Post normalization of Y hat values
		"""

		Yhatscojopred = (Yhatscojopred / sp.var(Yhatscojopred)) / max(1, n_cojo_selected_indices)

		Yhatscojopred_betainf = (Yhatscojopred_betainf / sp.var(Yhatscojopred_betainf)) / (m - n_cojo_selected_indices)
		Yhatscojopred_concatenated = (Yhatscojopred*alpha) + (Yhatscojopred_betainf * (1-alpha))





		validation_phen_noise =  stats.norm.rvs(0, sp.sqrt(1.0 - h2), size= n)
		validation_phen_noise = sp.sqrt((1.0 - h2) / sp.var(validation_phen_noise)) * validation_phen_noise

		validation_genetic_part = sp.dot(geno.T, betas)
		validation_genetic_part = sp.sqrt(h2 / sp.var(validation_genetic_part)) * validation_genetic_part

		validation_phen = validation_genetic_part + validation_phen_noise



		accuracy.append(stats.pearsonr(validation_phen, Yhatscojopred_concatenated)[0]**2)

	best_accuracy_index = accuracy.index(max(accuracy))
	best_alpha = alpha_list[best_accuracy_index]
	print "acc list", accuracy
	print "best alpha : ", best_alpha
	print "accuracy : ", max(accuracy)
	return best_alpha
def simulate_phenotypes(n, m, n_samples = 10,  genotype = None,  h2 = 0.5, p = 1.0, r2 = 0.9, m_ld_chunk_size = 100, p_threshold = 0.05, validation_set = None, alpha = None):
	#Simulate true betas
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
	betas_marg = (1.0 / n) * sp.dot(train_phen, traingeno.T)
	#simulate validation data or use existing data
	if alpha == None:
		alpha = estimate_alpha(n = 500, m = m,  h2 = 0.5, p = 1.0, r2 = 0.9, m_ld_chunk_size = 100, p_threshold = 0.05)

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

	#Simulate effect sizes
	noises = stats.norm.rvs(0,1,size=m)
	betainf = sp.zeros(m)
	if r2 == 0:
		beta_hats = betas + sp.sqrt(1.0/n) * noises
	else:
		#if ld
		C = sp.sqrt(((1.0)/n))*linalg.cholesky(sample_D)
		D_I = linalg.pinv(sample_D)
		betas_ld = sp.zeros(m)
		noises_ld = sp.zeros(m)
		for m_i in range(0,m,m_ld_chunk_size):
			m_end = m_i+m_ld_chunk_size
			betas_ld[m_i:m_end] = sp.dot(sample_D,betas[m_i:m_end])
			noises_ld[m_i:m_end]  = sp.dot(C.T,noises[m_i:m_end])
		beta_hats = betas_ld + noises_ld
		for m_i in range(0,m, m_ld_chunk_size):

			#calculate the ld corrected betaHats under the infinitesimal model
			m_end = m_i + m_ld_chunk_size
			A = ((m/h2) * sp.eye(m_ld_chunk_size) + (n/(1)) * sample_D)
			Ainv = linalg.pinv(A)
			betainf[m_i:m_end] = sp.dot(Ainv * n, beta_hats[m_i:m_end])

		ldDict = get_LDpred_ld_tables(traingeno, ld_radius=m_ld_chunk_size, h2=h2, n_training=n)
		betaLD = LDpred.ldpred_gibbs(beta_hats, start_betas = betainf, n = n, ld_radius = m_ld_chunk_size, p = p, ld_dict = ldDict["ld_dict"], h2 = h2)
		

	#estimate cojo corrected beta_hats
	cojo_beta_hats, cojopred_beta_hats, cojo_betainf, n_cojo_selected_indices = cojo.ml_iter(betas_marg, traingeno, ld_radius= m_ld_chunk_size, h2 = h2, p_threshold= p_threshold)

	cojopred_beta_hats = cojo_beta_hats[:]
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

	if p_threshold < 1:

		Z = n*beta_hats**2
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

	#for each genotype dataset
	for i in xrange(n_samples):


		#construct true validation phenotypes
		validation_phen_noise =  stats.norm.rvs(0, sp.sqrt(1.0 - h2), size= 2000)
		validation_phen_noise = sp.sqrt((1.0 - h2) / sp.var(validation_phen_noise)) * validation_phen_noise

		validation_genetic_part = sp.dot(geno[i].T, betas)
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

		Yhatscojopred[i] = (Yhatscojopred[i] / sp.var(Yhatscojopred[i])) / max(1, n_cojo_selected_indices)

		Yhatscojopred_betainf[i] = (Yhatscojopred_betainf[i] / sp.var(Yhatscojopred_betainf[i])) / (m - n_cojo_selected_indices)

		Yhatscojopred_concatenated.append((Yhatscojopred[i]*alpha) + (Yhatscojopred_betainf[i] * (1-alpha)))
		
		val_accuracy_cojopred.append(stats.pearsonr(validation_phen, Yhatscojopred_concatenated[i])[0]**2)
		print n_cojo_selected_indices
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

	return  accSMT, accPRS, accPval, accinf, accLDpred, acccojo, acccojopred, val_accuracy_cojopred

def printtable(filename, p, N, M, Ntraits, validationN = 5):
	with open(filename, 'w') as f:
		print >>f, 'N \t M \t P \t SMT \t PRS \t Pval \t LDinf \t LDpred \t cojo \t cojopred \n' 
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
						output = simulate_phenotypes(N[i], M[m], genotype = validation, n_samples = validationN,  h2 = 0.5, p = j, r2 = 0.9, m_ld_chunk_size = 100, p_threshold = 0.000001)

						print >>f, N[i],"\t",M[m],"\t",j,"\t",output[0],"\t",output[1], "\t", output[2], "\t", output[3], "\t", output[4], "\t", output[5], "\t", output[6], "\n"
if __name__ == "__main__":
	""
	p = [x*0.0002 for x in range(1,11)]
	p = p + [x*0.001 for x in range(1,11)]
	p = p + [x*0.01 for x in range(1,11)]
	p = p + [x*0.1 for x in range(1,11)]
	N = [600]
	M = [6000]
	printtable("effectofP_NM0.1", p = p, N = N, M = M, Ntraits = 5)
	print "0.1"
	N = [3000]
	printtable("effectofP_NM0.5", p = p, N = N, M = M, Ntraits = 5)
	print "0.5"
	N = [4800]
	printtable("effectofP_NM0.8", p = p, N = N, M = M, Ntraits = 5)
	print "0.8"
	N = [9000]
	printtable("effectofP_NM0.1.5", p = p, N = N, M = M, Ntraits = 5)
	print "1.5"
	
