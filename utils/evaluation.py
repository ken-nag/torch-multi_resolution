import mir_eval
import numpy as np
from pystoi import stoi
from pesq import pesq

def si_sdr(est, true):
    alpha = np.dot(true, est)/np.dot(true, true)
    target = alpha*true
    noise = target - est
    val = 10*np.log10(np.dot(target, target)/np.dot(noise, noise))
    return val


def sp_enhance_evals(est_source, clean_source, noisy_source, fs):
    est_source = est_source.cpu().clone().numpy()
    clean_source = clean_source.cpu().clone().numpy()
    noisy_source = noisy_source.cpu().clone().numpy()
    
    pesq_val = pesq(fs, clean_source, est_source, 'wb')
    stoi_val = stoi(clean_source, est_source, fs, extended=False)
    si_sdr_val = si_sdr(est_source, clean_source)
    noisy_si_sdr_val = si_sdr(noisy_source, clean_source)
    si_sdr_improvement = si_sdr_val -noisy_si_sdr_val
    return pesq_val, stoi_val, si_sdr_val, si_sdr_improvement

def mss_evals(est_source, clean, noisy):
    est_source = est_source.to('cpu').detach().numpy().copy()
    clean = clean.to('cpu').detach().numpy().copy()
    noisy = noisy.to('cpu').detach().numpy().copy()
    est_accompany = noisy - est_source
    true_accompany = noisy - clean
   
    
    true_len = est_source.shape[0]
    est_buff = np.zeros((2, true_len), dtype=np.float32)
    true_buff = np.zeros((2, true_len), dtype=np.float32)
    
    est_buff[0,:] = est_source[:]
    est_buff[1,:] = est_accompany[:]
    
    true_buff[0,:] = clean[:]
    true_buff[1,:] = true_accompany[:]
    sdr, sir, sar, perm =  mir_eval.separation.bss_eval_sources(true_buff, est_buff)
    si_sdr_val = si_sdr(est_source, clean)
    noisy_si_sdr_val = si_sdr(noisy, clean)
    si_sdr_improvement = si_sdr_val -noisy_si_sdr_val
    return sdr[0], sir[0], sar[0], si_sdr_val, si_sdr_improvement
     
    