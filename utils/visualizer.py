import numpy as np
import matplotlib.pyplot as plt

    
def show_TF_domein_result(train_loss, valid_loss, mixture, mask, estimate, true, vmin=-60, eps=1e-10):
    mixture = mixture.detach().clone().cpu().numpy()
    mask = mask.detach().clone().cpu().numpy()
    estimate = estimate.detach().clone().cpu().numpy()
    true = true.detach().clone().cpu().numpy()
    
    vmax = 20*np.log10(np.max(mixture+1e-8))-10
    vmin += vmax
    
    mask_max = 20*np.log10(np.max(mask+1e-8))-10
    mask_min = mask_max - 60
    
    fig, axes = plt.subplots(2,3, figsize=(12,8))

    axes[0,0].plot(train_loss)
    axes[0,0].set_title('train loss')
    axes[0,0].axis('auto')
    
    axes[0,1].plot(valid_loss)
    axes[0,1].set_title('valid loss')
    axes[0,1].axis('auto')
    
    axes[0,2].imshow(20*np.log10(np.flipud(mask)+eps), vmax=mask_max, vmin=mask_min, aspect="auto")
    axes[0,2].set_title('mask')
    
    axes[1,0].imshow(20*np.log10(np.flipud(estimate)+eps), vmax=vmax, vmin=vmin, aspect="auto") 
    axes[1,0].set_title('estimate')
    
    axes[1,1].imshow(20*np.log10(np.flipud(true)+eps), vmax=vmax, vmin=vmin, aspect="auto")
    axes[1,1].set_title('true')
    
    axes[1,2].imshow(20*np.log10(np.flipud(mixture)+eps), vmax=vmax, vmin=vmin, aspect="auto")
    axes[1,2].set_title('mixture')
    

        
    plt.tight_layout()
        
    plt.show()
    plt.close()
