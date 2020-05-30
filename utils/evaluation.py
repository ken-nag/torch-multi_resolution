import mir_eval
import numpy as np

def si_sdr():
    pass

def mss_evals(est_source, est_accompany, true_source, true_accompany):
    est_source = est_source.to('cpu').detach().numpy().copy()
    est_accompany = est_accompany.to('cpu').detach().numpy().copy()
    true_source = true_source.to('cpu').detach().numpy().copy()
    true_accompany = true_accompany.to('cpu').detach().numpy().copy()
    
    true_len = est_source.shape[0]
    est_buff = np.zeros((2, true_len), dtype=np.float32)
    true_buff = np.zeros((2, true_len), dtype=np.float32)
    
    est_buff[0,:] = est_source[:]
    est_buff[1,:] = est_accompany[:]
    
    true_buff[0,:] = true_source[:]
    true_buff[1,:] = true_accompany[:]
    sdr, sir, sar, perm =  mir_eval.separation.bss_eval_sources(true_buff, est_buff)
    return sdr[0], sir[0], sar[0]
     
    