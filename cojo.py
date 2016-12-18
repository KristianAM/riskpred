
import scipy as sp
from scipy import stats
from scipy import linalg
import phenotypes

def ml_iter(beta_hats, snps, ld_radius=100, h2 = 0.5, p_threshold=5*10e-8):
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
                print 'pval', ld_table[i]['pval']
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
            if len(ld_partners) >= 1:

                Xi= snps[i]
                Xpartners = snps[ld_partners]
                upb = sp.array(updated_beta_hats)[ld_partners]
                Di = sp.dot(Xi, Xpartners.T) / float(n_test)
                updated_beta_hats[i] = beta_hats[i] - sp.sum(sp.dot(Di, upb))
                cojo_updated_beta_hats[i] = 0

    #Remove cojosnps
    snps = sp.array(snps)
    newX = snps[non_selected_indices]
    assert newX.shape[0] + len(selected_indices) == len(beta_hats)
    newbetas = sp.array(updated_beta_hats)[non_selected_indices]

    m,n = newX.shape
    D = sp.dot(newX, newX.T) / float(n)
    betainf = sp.zeros(m)
    for m_i in range(0,m, ld_radius):
        #calculate the ld corrected betaHats under the infinitesimal model
        m_end = min(m_i + ld_radius, m)
        A = ((m/h2) * sp.eye(min(ld_radius, m_end-m_i)) + (n/(1)) * D[m_i:m_end, m_i:m_end])
        try:
            Ainv = linalg.pinv(A)
        except Exception(err_str):
            print err_str
            print linalg.pinv(D)
        betainf[m_i:m_end] = sp.dot(Ainv * n, newbetas[m_i:m_end])



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
    #     print cojo_beta_hats
    #     cojo_std = sp.sqrt(sp.var(cojo_beta_hats))
    # cojo_beta_hats_norm = (cojo_beta_hats / cojo_std) / len(selected_indices)
    # print "cojo snps", len(selected_indices)
    # print "difference", sp.absolute(sp.sum(cojo_beta_hats_norm**2) - sp.sum(betainf**2))
    
    # assert sp.absolute(sp.sum(cojo_beta_hats_norm**2) - sp.sum(betainf**2)) < 0.05

    # assert len(selected_indices) + len(non_selected_indices) == len(beta_hats), "error"

    # updated_beta_hats_conc = sp.zeros(len(beta_hats))
    # for i in range(len(beta_hats)):
    #     if i in selected_indices:
    #         index = selected_indices.index(i)
    #         updated_beta_hats_conc[i] = cojo_beta_hats_norm[index]
    #     else:
    #         index = non_selected_indices.index(i)
    #         updated_beta_hats_conc[i] = betainf[index]
    
    betainf_with_0 = sp.zeros(len(beta_hats))
    updated_beta_hats_with_0 = sp.zeros(len(beta_hats))
    betainf_with_0[non_selected_indices] = betainf

    updated_beta_hats_with_0 = cojo_updated_beta_hats
    cojo_updated_beta_hats = sp.array(cojo_updated_beta_hats)

    print len(updated_beta_hats)
    print len(betainf_with_0)
    return cojo_updated_beta_hats, betainf_with_0, len(selected_indices)

if __name__ == "__main__":
	test = phenotypes.simulate_phenotypes(200, 1000, n_samples = 2, p = 0.5, r2 = 0.9, h2 = 0.5, alpha = 0.5)
