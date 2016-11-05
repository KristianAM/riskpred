
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

    stop_boolean = True

    while stop_boolean == True:

        #Sort and select beta
        l = zip((sp.array(updated_beta_hats) ** 2).tolist(), range(m))
        l.sort(reverse=True)
        for beta_hat, beta_i in l:
            if not beta_i in selected_indices:
                beta_i_pval = stats.chi2.sf((beta_hat**2)*snps.shape[0], 1)
                if beta_i_pval <= p_threshold:
                    selected_indices.add(beta_i)
                    break
                if beta_i_pval > p_threshold:
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
                updated_beta = sp.dot(D_inv[0], bs)

                updated_beta_hats[i] = updated_beta
    cojo_updated_beta_hats = updated_beta_hats[:]
    selected_indices = list(selected_indices)
    non_selected_indices = []
    for i in range(m):
        if not i in selected_indices:
            d = ld_table[i]
            non_selected_indices.append(i)

            ld_partners = d['ld_partners'][1:]

            Xi= snps[i]
            Xpartners = snps[ld_partners]
            upb = sp.array(updated_beta_hats)[ld_partners]
            Di = sp.dot(Xi, Xpartners.T) / float(n_test)
            updated_beta_hats[i] = beta_hats[i] - sp.sum(sp.dot(Di, upb))
            cojo_updated_beta_hats[i] = 0

    #Remove cojosnps

    newX = sp.delete(snps, (selected_indices), axis = 0)
    assert newX.shape[0] + len(selected_indices) == len(beta_hats)
    newbetas = [0 if x in selected_indices else x for x in updated_beta_hats]

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
    betainf_with_0 =  [0 for x in range(len(beta_hats))]
    updated_beta_hats_with_0 = [0 for x in range(len(beta_hats))]
    for i in range(len(beta_hats)):
        if i in selected_indices:
            index = selected_indices.index(i)
            betainf_with_0[i] = 0
            updated_beta_hats_with_0[i] = updated_beta_hats[index]
        else:
            index = non_selected_indices.index(i)
            betainf_with_0[i] = betainf[index]
            updated_beta_hats_with_0[i] = 0
    cojo_updated_beta_hats = sp.array(cojo_updated_beta_hats)

    print len(updated_beta_hats)
    print len(betainf_with_0)
    return cojo_updated_beta_hats, updated_beta_hats, betainf_with_0, len(selected_indices)


if __name__ == "__main__":
	test = phenotypes.simulate_phenotypes(500, 1000, n_samples = 10, p = 0.3, r2 = 0.9, p_threshold=0.5, h2 = 0.5)
