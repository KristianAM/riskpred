
import scipy as sp
from scipy import stats
from scipy import linalg
import phenotypes
def ml_iter(beta_hats, snps, ld_radius=100, h2 = 0.5, p_threshold=0.000001):
    """
    Yang et al. iterative scheme.
    
    Idea:
    - While # selected betas is < threshold
    -     Sort betas
    -     Pick the largest beta that hasn't been selected. 
    -     For each marker in window around selected marker:
    -         Update LD matrix
    -         Invert LD matrix
    -         Update beta

    """

    Z = snps.shape[0]*beta_hats**2
    pval = []
    indices = []
    non_selected_indices = []
    for val in Z:
        pval.append(stats.chi2.sf(val, 1))
    for j in range(len(pval)):
        if pval[j] < p_threshold:
            indices.append(j)


    #Ordering the beta_hats, and storing the order
    m = len(beta_hats)
    beta_hats = beta_hats.tolist()
    ld_table = {}
    for i in range(m):
        ld_table[i] = {'ld_partners':[i], 'beta_hat_list':[beta_hats[i]], 'D' : sp.array([1.0]), 'D_inv':sp.array([1.0])}
    snps = sp.array(snps)
    n_test = len(snps[0])
    snps = sp.array(snps)
    max_num_selected = len(indices)
    selected_indices = set()
    updated_beta_hats = beta_hats[:]

    while len(selected_indices) < max_num_selected:
        #Sort and select beta
        l = zip((sp.array(updated_beta_hats) ** 2).tolist(), range(m))
        l.sort(reverse=True)
        for beta_hat, beta_i in l:
            if not beta_i in selected_indices:
                selected_indices.add(beta_i)
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
                updated_beta = sp.dot(D_inv[0], bs)

                updated_beta_hats[i] = updated_beta
    cojo_updated_beta_hats = updated_beta_hats[:]
    selected_indices = list(selected_indices)
    non_selected_indices = []
    for i in range(m):
        if not i in selected_indices:
            d = ld_table[i]
            non_selected_indices.append(i)
            #use updated betas for the partners
            #CV relevant courses + project + reference Mikkel & bjarni
            ld_partners = d['ld_partners'][1:]

            #super slow
            Xi= snps[i]
            Xpartners = snps[ld_partners]
            upb = sp.array(updated_beta_hats)[ld_partners]
            Di = sp.dot(Xi, Xpartners.T) / float(n_test)
            updated_beta_hats[i] = beta_hats[i] - sp.sum(sp.dot(Di, upb))
            cojo_updated_beta_hats[i] = 0

    #Remove cojosnps

    newX = sp.delete(snps, (selected_indices), axis = 0)
    newbetas = [0 if x in selected_indices else x for x in updated_beta_hats]

    m,n = newX.shape
    D = sp.dot(newX, newX.T) / float(n)
    betainf = sp.zeros(m)
    for m_i in range(0,m, ld_radius):
        #calculate the ld corrected betaHats under the infinitesimal model
        m_end = min(m_i + ld_radius, m)
        A = ((m/h2) * sp.eye(min(ld_radius, m_end-m_i)) + (n/(1)) * D[m_i:m_end, m_i:m_end])
        Ainv = linalg.pinv(A)
        betainf[m_i:m_end] = sp.dot(Ainv * n, newbetas[m_i:m_end])



    updated_beta_hats_conc = sp.zeros(len(beta_hats))
    cojo_beta_hats = sp.array(updated_beta_hats)[selected_indices]
    for i in range(len(beta_hats)):
        if i in selected_indices:
            index = selected_indices.index(i)
            updated_beta_hats_conc[i] = cojo_beta_hats[index]
        if i in non_selected_indices:
            index = non_selected_indices.index(i)
            updated_beta_hats_conc[i] = betainf[index]



    cojo_updated_beta_hats = sp.array(cojo_updated_beta_hats)


    return cojo_updated_beta_hats, updated_beta_hats_conc


def return_indices_of_a(a, b):
  b_set = set(b)
  return [i for i, v in enumerate(a) if v in b_set]

if __name__ == "__main__":
	test = phenotypes.simulate_phenotypes(10000, 1000, n_samples = 5, p = 0.5, r2 = 0.9, p_threshold=0.5, h2 = 0.5)
