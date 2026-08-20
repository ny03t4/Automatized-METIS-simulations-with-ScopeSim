import numpy as np


"""EMPIRICAL SNR FUNCTION"""

def mean_Variance(rate, t):
    N = np.size(rate)
    return np.sum(rate)*t/N
    
    
def SNR_EMPIRICAL(bgsub_readout, sky_readout_data, dit, ndit, subarray_size, distance_noise_subarray, chop_nod): 
    """Calculate the SNR with empirical method

    Args:
        bgsub_readout (array): background subtracted readout multiplied by the GAIN /!\
        sky_readout_data (array): sky readout data
        dit (float): dit
        ndit (float): ndit /!\ *4 in case of chop_nod == True
        subarray_size (int): size of the subarray
        distance_noise_subarray (int): distance from the center of the "noise" subarray
        chop_nod (Boolean): True if chop_nod included, False otherwise

    Returns:
        array, float: SNR map, mean SNR on the subarray
    """    
    src = (bgsub_readout + sky_readout_data)
    
    # Subarray def
    screen_center_readout = int(np.shape(src)[0]/2)
    # (y_start, y_end, x_start, x_end)
    limits_source = subarray_size
    signal_coords_readout = (screen_center_readout-limits_source, 
                             screen_center_readout+limits_source, 
                             screen_center_readout-limits_source, 
                             screen_center_readout+limits_source)
    
    
    noise_coords = (screen_center_readout-limits_source, 
                    screen_center_readout+limits_source, 
                    distance_noise_subarray-limits_source, 
                    distance_noise_subarray+limits_source)

    #Extract data from 
    bg_src = src[noise_coords[0]:noise_coords[1], noise_coords[2]:noise_coords[3]] 
    signal = src[signal_coords_readout[0]:signal_coords_readout[1], 
                 signal_coords_readout[2]:signal_coords_readout[3]] 

    if chop_nod == True: # because background already substracted
        bg_src = np.zeros(np.shape(signal)) 
        sky_cropped = np.zeros(np.shape(signal))
    else: 
        sky_cropped = sky_readout_data[signal_coords_readout[0]:signal_coords_readout[1], 
                                        signal_coords_readout[2]:signal_coords_readout[3]] # [ADU]
        
    t = dit*ndit   
    mean_signal = np.nanmean(signal)
    mean_bg = np.nanmean(bg_src)
    mean_var_signal = mean_Variance(signal, t)
    mean_var_bg = mean_Variance(bg_src,t)
    mean_sky_cropped = np.nanmean(sky_cropped)
     
    
    snr_emp_mean = t*(mean_signal-mean_bg)/np.sqrt(mean_var_signal + mean_var_bg)
    snr_emp_map = t*(signal-bg_src)/np.sqrt(t*signal+t*bg_src)
    
    return snr_emp_map, snr_emp_mean


"""THEORETICAL SNR FUNCTION"""

ron = metis.cmds["!DET.readout_noise"] # [e-/integration^(1/2)]
darkCurrent = metis.cmds['!DET.dark_current'] # [e-/s]
gain = metis.cmds["!DET.gain"] # [e-/ADU]

if mode == "img_n":
    QE = 0.8 # [-]
if mode == "img_lm":
    QE =  0.8574456 # [-]
        
def SNR_TH(bgsub_psfsub_image, bgsub_image, bgsub_psf, bg, n_pix, ron, darkCurrent, dit, ndit):
    """Calculation SNR following theoretical formula

    Args:
        bgsub_psfsub_signal (array): background and psf subtracted image /!\ MUTLIPLIED BY THE GAIN (readout image) OR QE (implane image)
        bgsub_signal (array): background subtracted image /!\ MUTLIPLIED BY THE GAIN (readout image) OR QE (implane image)
        bgsub_psf (array): background subtracted psf image /!\ MUTLIPLIED BY THE GAIN (readout image) OR QE (implane image)
        bg (array): background image /!\ MUTLIPLIED BY THE GAIN (readout image) OR QE (implane image)
        n_pix (int): number of pixels on the screen
        ron (float): Read-out-noise
        darkCurrent (float): Darkcurrent
        dit (float): dit
        ndit (float): ndit /!\ *4 in case of chop_nod == True

    Returns:
        array: SNR map
    """
    ron_per_pixel = ron/n_pix  # [e-/integration^(1/2)/pxl]
    darkCurrent_per_pixel = darkCurrent/n_pix # [e-/s/pxl]
    t = dit*ndit # [s]
    snr_th_map = t * bgsub_psfsub_image / np.sqrt(t*bgsub_image + t*bgsub_psf + bg*t + n_pix*(t*darkCurrent_per_pixel + ndit*ron_per_pixel**2))
    
    
    return snr_th_map 