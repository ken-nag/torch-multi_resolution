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
     
    