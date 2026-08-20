import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits
from astropy import units as u
import scopesim as sim
import os 

from variables import *

#===========================================
#                CONVERSION
#===========================================


def pc_to_mas(D, S):
    """Convert the size of an object in pc to its angular size in mas using the distance to the source

    Args:
        D (Quantity): Distance between the source and the detector [length]
        S (Quantity): Size of the object [length]

    Returns:
        Quantity: Angular size of the object in mas
    """
    D = D.to(u.pc)
    S = S.to(u.pc)
    mas = 2*np.arctan(S/2/D)
    mas = mas.to(u.mas)
    return mas


def mas_to_pc(theta, D):
    """Convert the angular size of an object in mas to its size in pc using the distance to the source

    Args:
        theta (Quantity): Angular size [angle]
        D (Quantity): Distance between the source and the detector [length]

    Returns:
        Quantity: Size of the object in pc
    """
    D = D.to(u.pc)
    theta = theta.to(u.radian)
    pc = 2*D*np.tan(theta/2)
    pc = pc.to(u.pc)
    return pc


def mean_Variance(rate, t):
    N = np.size(rate)
    return np.sum(rate)*t/N

#===========================================
#            Elliptical annulus
#===========================================

def elliptical_annulus_mean_rotated(image, a, b, CDELT, thickness, theta, windows_lim):
    """
    Compute the mean value of pixels within a rotated elliptical annulus.

    Parameters:
    - image: 2D numpy array
    - a: semi-major axis of inner ellipse [mas]
    - b: semi-minor axis of inner ellipse [mas]
    - thickness: thickness of the annulus
    - center: tuple (cy, cx)
    - theta: rotation angle in radians (counterclockwise)

    Returns:
    - mean value within the annulus
    """
    # conversion
    a = a.to(u.mas)
    b = b.to(u.mas)
    CDELT = CDELT.to(u.mas)
    theta = theta.to(u.radian)
    windows_lim = windows_lim.to(u.mas)
    
    cy, cx = np.shape(image)
    cx /= 2
    cy /= 2
    # Inner and Outer ellipse axes
    a = int(a.value/CDELT.value) # to obtain the number of pixels
    b = int(b.value/CDELT.value)
    a_outer = a + thickness
    b_outer = b + thickness

    # Coordinate grid
    y, x = np.indices(image.shape)

    # Shift coordinates to center
    x_shift = x - cx
    y_shift = y - cy

    # Precompute trig
    cos_t = np.cos(theta.value)
    sin_t = np.sin(theta.value)

    # Rotate coordinates
    x_rot = x_shift * cos_t + y_shift * sin_t
    y_rot = -x_shift * sin_t + y_shift * cos_t

    # Ellipse equations
    inner = (x_rot**2) / a**2 + (y_rot**2) / b**2
    outer = (x_rot**2) / a_outer**2 + (y_rot**2) / b_outer**2

    # Annulus mask
    mask = (outer <= 1) & (inner >= 1)

    
   # PLOT
    masked_image = np.where(mask, image, np.nan)
    
    # change axis to mas instead of pxl
    ref_pt1 = np.shape(image)[0]/2
    ref_pt2 = np.shape(image)[1]/2
    extent_array = [-ref_pt1*CDELT.value, ref_pt1*CDELT.value, -ref_pt2*CDELT.value, ref_pt2*CDELT.value]
    
    plt.imshow(image, origin='lower', norm='log', extent=extent_array)
    plt.imshow(masked_image, cmap='Greys', origin='lower', extent=extent_array)
    #plt.colorbar(image)
    plt.title('Image with mask (with log norm)')
    plt.xlabel('[mas]')
    plt.ylabel('[mas]')
    plt.xlim(-windows_lim.value, windows_lim.value)
    plt.ylim(-windows_lim.value, windows_lim.value)
    plt.show()
    
    # return
    if not np.any(mask):
        return np.nan

    return image[mask].mean()

#===========================================
#                FITS FILE
#===========================================

def create_fits(array, header, variable_name, object_name, mode, exptime, chop_nod_status, psfsub_status):
    new_header= header.copy()
    new_header['BITPIX'] = -64
    new_header['NAXIS1'] = np.shape(array)[0]
    new_header['NAXIS2'] = np.shape(array)[1]
    new_header['CRPIX1'] = np.shape(array)[0]/2
    new_header['CRPIX2'] = np.shape(array)[1]/2
    new_header['object'] = object_name
    exptime = exptime.to(u.minute)
    
    directory_name = '../outputs/'+mode+'/'+object_name
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    
    file = '../outputs/'+mode+'/'+object_name+'/'+str(exptime.value)+'min_'+chop_nod_status+'_'+psfsub_status+'_'+variable_name+'.fits'
    hdu = fits.PrimaryHDU(array, header=new_header)
    hdu.writeto(file, overwrite=True)
    return 0

#===========================================
#            CIRCULAR GAUSSIAN
#===========================================

def Circular_Gaussian_profile(FWHM, flux, pixel_size, D_source, example_file, mode, object_name, creating_FITs):
    """Generate a 2D circular Gaussian profile and create a FITS file.

    Args:
        FWHM (Quantity): FWHM of the source
        flux (Quantity): Total flux of the source received by the detector
        pixel_size (Quantity): Size of the pixel in pc

    Returns:
        Array(float): 2D array containing the 2D circular Gaussian profile
    """
    FWHM = FWHM.to(u.pc)
    FWHM = FWHM.value
    pixel_size = pixel_size.to(u.pc)
    #pixel_size = pixel_size.value
    flux = flux.to(u.Jy)
    flux = flux.value
    
    sigma = FWHM/(2* np.sqrt(2*np.log(2)))/pixel_size.value
    grid_length = int(10*sigma)
    
    if grid_length % 2 == 0:
        grid_length += 1
    Grid_size = (grid_length, grid_length)

    x = [i for i in range(Grid_size[0])]
    y = [i for i in range(Grid_size[1])]
    
    # The source is centered at the center of the grid
    x_c = int(Grid_size[0]/2) 
    y_c = int(Grid_size[1]/2)
    I = 1
    G = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            G[i,j] = I*np.exp(-0.5*(((x[i]-x_c)/sigma)**2 + ((y[j]-y_c)/sigma)**2))
    
    # To normalize the gaussian following the value of the total flux of the source
    somme = np.sum(G)
    ratio = flux/somme 
    G *= ratio
    
    # Creation of a FITS file to store the Gaussian
    if creating_FITs == True:
        example_data, example_header = fits.getdata(example_file, header=True)
        new_header= example_header.copy()

        CDELT_circinus = pc_to_mas(D_source, pixel_size).to(u.degree)
        new_header['BITPIX'] = -64
        new_header['NAXIS1'] = np.shape(G)[0]
        new_header['NAXIS2'] = np.shape(G)[1]
        new_header['CRPIX1'] = np.shape(G)[0]/2
        new_header['CRPIX2'] = np.shape(G)[1]/2
        new_header["CDELT1"] = CDELT_circinus.value
        new_header["CDELT2"] = CDELT_circinus.value
        new_header['object'] = object_name
        
        directory_name = '../outputs/'+mode+'/'+object_name
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)
            
        file = '../outputs/'+mode+'/'+object_name+'/MODEL_Circular_Gaussian.fits'
        hdu = fits.PrimaryHDU(G, header=new_header)
        hdu.writeto(file, overwrite=True)
    
    return G


#===========================================
#            ELLIPTICAL GAUSSIAN
#===========================================

def Elliptical_Gaussian_profile(FWHM_x, FWHM_y, theta, flux, pixel_size, D_source, grid_length_factor, example_file, mode, object_name, creating_FITs):
    """Generate a 2D Elliptical Gaussian profile.

    Args:
        FWHM_x (Quantity): FWHM of the source in x axis
        FWHM_y (Quantity): FWHM of the source in y axis
        flux (Quantity): Total flux of the source received by the detector
        pixel_size (Quantity): Size of the pixel in pc

    Returns:
        Array(float): 2D array containing the 2D elliptical Gaussian profile
    """
    FWHM_x = FWHM_x.to(u.pc)
    FWHM_x = FWHM_x.value
    FWHM_y = FWHM_y.to(u.pc)
    FWHM_y = FWHM_y.value
    theta = theta.to(u.radian)
    theta = theta.value 
    
    pixel_size = pixel_size.to(u.pc)
    #pixel_size = pixel_size.value
    flux = flux.to(u.Jy)
    flux = flux.value
    
    sigma_x = FWHM_x/(2* np.sqrt(2*np.log(2)))/pixel_size.value
    sigma_y = FWHM_y/(2* np.sqrt(2*np.log(2)))/pixel_size.value
    grid_length = int(grid_length_factor*np.max([sigma_x, sigma_y]))
    
    if grid_length % 2 == 0:
        grid_length += 1
    Grid_size = (grid_length, grid_length)

    x = [i for i in range(Grid_size[0])]
    y = [i for i in range(Grid_size[1])]
    
    # The source is centered at the center of the grid
    x_c = int(Grid_size[0]/2) 
    y_c = int(Grid_size[1]/2)
    I = 1
    G = np.zeros((len(x), len(y)))
    a = np.cos(theta)**2/(2*sigma_x**2)+np.sin(theta)**2/(2*sigma_y**2)
    c = np.cos(theta)**2/(2*sigma_y**2)+np.sin(theta)**2/(2*sigma_x**2) 
    b = -np.cos(theta)*np.sin(theta)/(2*sigma_x**2)+np.cos(theta)*np.sin(theta)/(2*sigma_y**2)
    
    for i in range(len(x)):
        for j in range(len(y)):
            G[i,j] = I*np.exp(-(a*(x[i]-x_c)**2 + 2*b*(x[i]-x_c)*(y[j]-y_c) + c*(y[j]-y_c)**2))
    
    # To normalize the gaussian following the value of the total flux of the source
    somme = np.sum(G)
    ratio = flux/somme 
    G *= ratio

    # Creation of a FITS file to store the Gaussian
    if creating_FITs == True:
        example_data, example_header = fits.getdata(example_file, header=True)
        new_header= example_header.copy()

        CDELT_circinus = pc_to_mas(D_source, pixel_size).to(u.degree)
        new_header['BITPIX'] = -64
        new_header['NAXIS1'] = np.shape(G)[0]
        new_header['NAXIS2'] = np.shape(G)[1]
        new_header['CRPIX1'] = int(np.shape(G)[0]/2)
        new_header['CRPIX2'] = int(np.shape(G)[1]/2)
        new_header["CDELT1"] = CDELT_circinus.value
        new_header["CDELT2"] = CDELT_circinus.value
        new_header['object'] = object_name

        directory_name = '../outputs/'+mode+'/'+object_name
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)
            
        file = '../outputs/'+mode+'/'+object_name+'/MODEL_Elliptical_Gaussian.fits'
        hdu = fits.PrimaryHDU(G, header=new_header)
        hdu.writeto(file, overwrite=True)
    
    return G

#===========================================
#        1 pt source + ELLIPTICAL GAUSSIAN
#===========================================
def create_model_3compo(FWHM_polar_x, FWHM_polar_y, flux_polar, theta_polar,
                        FWHM_disk_x, FWHM_disk_y, flux_disk, theta_disk,
                        FWHM_unres, flux_unres, 
                        D_source, 
                        pixel_size, grid_length_factor, d_offset, 
                        example_file, 
                        mode, object_name, creating_FITs):
    
    G_unres = Circular_Gaussian_profile(FWHM_unres, 
                                        flux_unres, 
                                        pixel_size, D_source, example_file, mode, object_name, False)
    
    G_polar = Elliptical_Gaussian_profile(FWHM_polar_x, FWHM_polar_y, 
                                          theta_polar, flux_polar, 
                                          pixel_size, D_source,  grid_length_factor, example_file, mode, object_name, False)
    
    G_disk = Elliptical_Gaussian_profile(FWHM_disk_x, FWHM_disk_y, 
                                         theta_disk, flux_disk, 
                                         pixel_size, D_source, grid_length_factor, example_file, mode, object_name, False)
    d_center_polar = int(np.shape(G_polar)[0]/2+ d_offset) 
        
    # Adding G_disk
    half_length_disk = int(np.shape(G_disk)[0]/2)

    for i in range(np.shape(G_disk)[0]):
        for j in range(np.shape(G_disk)[1]):
            G_polar[d_center_polar-half_length_disk + i, d_center_polar-half_length_disk + j] += G_disk[i,j]
            
    # Adding G_unres
    if flux_unres != 0:
        half_length_unres = int(np.shape(G_unres)[0]/2)
        
        for i in range(np.shape(G_unres)[0]):
            for j in range(np.shape(G_unres)[1]):
                G_polar[d_center_polar-half_length_unres + i, d_center_polar-half_length_unres + j] += G_unres[i,j]
            
    
    
    # Creation of a FITS file to store the Gaussian
    if creating_FITs == True:
        example_data, example_header = fits.getdata(example_file, header=True)
        new_header= example_header.copy()

        CDELT = pc_to_mas(D_source, pixel_size).to(u.degree)
        new_header['BITPIX'] = -64
        new_header['NAXIS1'] = np.shape(G_polar)[0]
        new_header['NAXIS2'] = np.shape(G_polar)[1]
        new_header['CRPIX1'] = int(np.shape(G_polar)[0]/2)
        new_header['CRPIX2'] = int(np.shape(G_polar)[1]/2)
        new_header["CDELT1"] = CDELT.value
        new_header["CDELT2"] = CDELT.value
        new_header['object'] = object_name

        directory_name = '../outputs/'+mode+'/'+object_name
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)
            
        file = '../outputs/'+mode+'/'+object_name+'/MODEL_3compo.fits'
        hdu = fits.PrimaryHDU(G_polar, header=new_header)
        hdu.writeto(file, overwrite=True)
    
    
    # plot 
    ref_pt1 = np.shape(G_polar)[0]/2
    ref_pt2 = np.shape(G_polar)[1]/2
    CDELT = CDELT.to(u.mas)
    extent_array = [-ref_pt1*CDELT.value, ref_pt1*CDELT.value, -ref_pt2*CDELT.value, ref_pt2*CDELT.value]
    plt.imshow(G_polar, origin='lower', extent=extent_array)
    plt.colorbar(label='[Jy]')
    plt.title('Input model')
    plt.savefig('../outputs/'+mode+'/'+object_name+'/MODEL_3compo.pdf')
    plt.show()
    
    return G_polar



#===========================================
#             METIS OBSERVATION
#===========================================
def METIS_observation(file, chop_nod_offset, metis, dit, ndit, chop_nod):
    """Perform the METIS observation

    Args:
        file (string): file path
        metis (function): metis function
        dit (float): dit
        ndit (float): ndit
        chop_nod (boolean): True if chop & nod included

    Returns:
        list: implane_data, readout_data, header_implane header_readout
    """
    if file == 'sky':
        metis.observe()
    else:
        with fits.open(file) as hdul:
                image = sim.Source(image_hdu=hdul[0])
        metis.observe(image)

    implane_data = metis.image_planes[0].data # [ph/s]
    header_implane = metis.image_planes[0].header
    
    if chop_nod == True:
        chop_nod_offset = chop_nod_offset.to(u.arcsec)
        metis['chop_nod'].include = True
        metis["chop_nod"].meta["chop_offsets"] = [chop_nod_offset.value, 0]
        metis["chop_nod"].meta["nod_offsets"] = [0, chop_nod_offset.value]
        print("Chopping:", np.array([chop_nod_offset.value, 0]))
        print("Nodding: ", np.array([0, chop_nod_offset.value]))
        
    readout  = metis.readout(dit=dit, ndit=ndit)[0] 
    readout_data = readout[1].data # [ADU]
    header_readout = readout[1].header
    return implane_data, readout, readout_data, header_implane, header_readout

#===========================================
#                CHOP NOD
#===========================================


def chop_nod_fct(metis_readout, chop_nod_offset, dit, ndit, mode):
    image = metis_readout[1].data
    metis_readout[0].header['DIT'] = dit
    metis_readout[0].header['NDIT'] = ndit 
    metis_readout[0].header['EXPTIME'] = dit * ndit
    
    if mode == "img_n":
        pixel_scale = N_band_pixel_scale
    if mode == "img_lm":
        pixel_scale = L_band_pixel_scale
        
    #chop_nod_offset = 3*u.arcsec
    chop_nod_offset = chop_nod_offset.to(u.mas)
    
    shift = int(chop_nod_offset.value / pixel_scale.value)
    quadrant_size = int(shift/2)
    screen_center = int(np.shape(image)[0]/2)
    border_min = int(screen_center - quadrant_size)
    border_max = int(screen_center+shift + quadrant_size)

    xmin, xmax = border_min, int(border_min+(border_max-border_min)/2)
    ymin, ymax = xmin, xmax

    beam_1 = image[ymin:ymax, xmin:xmax]
    beam_2 = image[ymin:ymax, xmin+shift:xmax+shift]
    beam_3 = image[ymin+shift:ymax+shift, xmin:xmax]
    beam_4 = image[ymin+shift:ymax+shift, xmin+shift:xmax+shift]
    bgsub_readout = beam_1 - beam_2 - beam_3 + beam_4 
    return bgsub_readout 

#===========================================
#              EMPIRICAL SNR 
#===========================================


def SNR_EMPIRICAL(bgsub_readout, sky_readout_data, dit, ndit, subarray_size, distance_noise_subarray, chop_nod): 
    """Calculate the SNR with empirical method

    Args:
        bgsub_readout (array): background subtracted readout !!! MUTLIPLIED BY THE GAIN
        sky_readout_data (array): sky readout data
        dit (float): dit
        ndit (float): ndit  !!! *4 in case of chop_nod == True
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

#===========================================
#                SNR TH
#===========================================

def SNR_TH(bgsub_psfsub_image, bgsub_image, bgsub_psf, bg, n_pix, ron, darkCurrent, dit, ndit):
    """Calculation SNR following theoretical formula

    Args:
        bgsub_psfsub_signal (array): background and psf subtracted image !!! MUTLIPLIED BY THE GAIN (readout image) OR QE (implane image)
        bgsub_signal (array): background subtracted image !!! MUTLIPLIED BY THE GAIN (readout image) OR QE (implane image)
        bgsub_psf (array): background subtracted psf image !!! MUTLIPLIED BY THE GAIN (readout image) OR QE (implane image)
        bg (array): background image !!! MUTLIPLIED BY THE GAIN (readout image) OR QE (implane image)
        n_pix (int): number of pixels on the screen
        ron (float): Read-out-noise
        darkCurrent (float): Darkcurrent
        dit (float): dit
        ndit (float): ndit  !!! *4 in case of chop_nod == True

    Returns:
        array: SNR map
    """
    ron_per_pixel = ron/n_pix  # [e-/integration^(1/2)/pxl]
    darkCurrent_per_pixel = darkCurrent/n_pix # [e-/s/pxl]
    t = dit*ndit # [s]
    snr_th_map = t * bgsub_psfsub_image / np.sqrt(t*bgsub_image + t*bgsub_psf + bg*t + n_pix*(t*darkCurrent_per_pixel + ndit*ron_per_pixel**2))
    
    
    return snr_th_map 

#===========================================
#             FULL SNR PROCESS
#===========================================

def SNR_calculation(object_name, file_src,  
                    mode, exptime, window_lim, file_full_src='nothing',
                    chop_nod=False, chop_nod_offset=3*u.arcsec,
                    psf_subtraction=False, psf_file='nothing', 
                    SNR_implane_plot=False, SNR_readout_plot=False, 
                    contour_plot=False,
                    creating_FITs=True): 
    
    if mode == "img_n":
        band = "N-band"
    if mode == "img_lm":
        band = "L-band"
    
    # METIS set up
    cmd = sim.UserCommands(use_instrument="METIS", set_modes=[mode], properties={"!OBS.exptime": exptime.value, "!OBS.dit": None, "!OBS.ndit": None})
    metis = sim.OpticalTrain(cmd)
    
    # Sky observation
    print('======================= SKY OBSERVATION =======================')
    metis.observe()      # blank sky
    
    sky_implane = metis.image_planes[0].data # [ph/s]
    sky_readout = metis.readout(exptime=exptime.value)[0]
    sky_readout_data = sky_readout[1].data
    dit, ndit = metis.cmds["!OBS.dit"], metis.cmds["!OBS.ndit"]
    
    # PSF subtraction
    if psf_subtraction == True:
        print('======================= PSF OBSERVATION =======================')
        psf_implane, psf_readout, psf_readout_data, header_psf_implane, header_psf_readout = METIS_observation(psf_file, chop_nod_offset, metis, dit, ndit, chop_nod)
        if chop_nod == True:
            psf_chop_nod = chop_nod_fct(psf_readout, chop_nod_offset, dit, ndit*4, mode)
            
    # Object observation
    print('======================= OBJECT OBSERVATION =======================')
    src_implane, src_readout, src_readout_data, header_src_implane, header_src_readout  = METIS_observation(file_src, chop_nod_offset, metis, dit, ndit, chop_nod)
    if chop_nod == True:
        bgsub_readout = chop_nod_fct(src_readout, chop_nod_offset, dit, ndit*4, mode) 
        
        
    # Background subtraction 
    if chop_nod == False and psf_subtraction == False:
        bgsub_implane = src_implane - sky_implane
        bgsub_readout = src_readout_data - sky_readout_data 
        
        bgsub_psf_implane = np.zeros(np.shape(bgsub_implane))
        bgsub_psf_readout = np.zeros(np.shape(bgsub_readout))
        
    if chop_nod == False and psf_subtraction == True:
        bgsub_implane = src_implane - psf_implane
        bgsub_readout = src_readout_data - psf_readout_data 
        
        bgsub_psf_implane = psf_implane - sky_implane
        bgsub_psf_readout = psf_readout_data - sky_readout_data
        
    if chop_nod == True and psf_subtraction == False:
        bgsub_implane = src_implane - sky_implane
        
        bgsub_psf_implane = np.zeros(np.shape(bgsub_implane))
        bgsub_psf_readout = np.zeros(np.shape(bgsub_readout))
    
    if chop_nod == True and psf_subtraction == True:
        bgsub_implane = src_implane - psf_implane
        bgsub_readout -= psf_chop_nod
        
        bgsub_psf_implane = psf_implane - sky_implane
        bgsub_psf_readout = psf_chop_nod
        
    # Full source observation 
    if file_src ==  file_full_src or file_full_src == 'nothing':
        bgsub_implane_full_src = src_implane - sky_implane
        bgsub_readout_full_src = src_readout_data - sky_readout_data 
        if chop_nod == True:
            bgsub_readout_full_src = chop_nod_fct(src_readout, chop_nod_offset, dit, ndit*4, mode) 
    
    else:
        print('======================= FULL SOURCE OBSERVATION =======================')
        full_src_implane, full_src_readout, full_src_readout_data, header_full_src_implane, header_full_src_readout  = METIS_observation(file_full_src, chop_nod_offset, metis, dit, ndit, chop_nod)
        
        bgsub_implane_full_src = full_src_implane - sky_implane
        if chop_nod == True:
            bgsub_readout_full_src = chop_nod_fct(full_src_readout, chop_nod_offset, dit, ndit*4, mode) 
        if chop_nod == False:
            bgsub_readout_full_src = full_src_readout_data - sky_readout_data
           
    # Sky trimming (because chop & nod reduces the size of the image)    
    if chop_nod == True:
        length = np.shape(bgsub_readout)[0]
        half_length = int(length/2)
        sky_readout_data = sky_readout_data[length-half_length :length+half_length,
                                            length-half_length :length+half_length]


    """SNR CALCULATION"""
    # Detector information 
    n_pix_implane = np.shape(bgsub_implane)[0]**2
    n_pix_readout = np.shape(bgsub_readout)[0]**2
    #metis.cmds["!DET"]
    ron = metis.cmds["!DET.readout_noise"] # [e-/integration^(1/2)]
    darkCurrent = metis.cmds['!DET.dark_current'] # [e-/s]
    gain = metis.cmds["!DET.gain"] # [e-/ADU]
    
    if mode == "img_n":
        QE = 0.8 # [-]
    if mode == "img_lm":
        QE =  0.8574456 # [-]
        
    
    if chop_nod == True:
        ndit *= 4
    # Theoretical SNR map (implane)
    snr_th_implane = SNR_TH(bgsub_implane*QE, bgsub_implane_full_src*QE, bgsub_psf_implane*QE, sky_implane*QE, n_pix_implane, ron, darkCurrent, dit, ndit)
    
    # Theoretical SNR map (readout)
    snr_th_readout = SNR_TH(bgsub_readout*gain, bgsub_readout_full_src*gain, bgsub_psf_readout*gain, sky_readout_data*gain, n_pix_readout, ron, darkCurrent, dit, ndit)
    
    # Estimation of SNR from the data
    # snr_emp_map, snr_emp_mean, snr_th_cropped = SNR_DATA(bgsub_readout*gain, sky_readout_data*gain, dit, ndit, subarray_size, distance_noise_subarray, chop_nod)
    
    
    """" PLOTS """
    # change axis to mas instead of pxl
    ref_pt1 = np.shape(bgsub_readout)[0]/2
    ref_pt2 = np.shape(bgsub_readout)[1]/2
    CDELT1 = header_src_readout['CDELT1'] *u.arcsec
    CDELT1 = CDELT1.to(u.mas)
    extent_array = [-ref_pt1*CDELT1.value, ref_pt1*CDELT1.value, -ref_pt2*CDELT1.value, ref_pt2*CDELT1.value]
    
    # plots
    exptime_plot = exptime.to(u.minute)
    
    if chop_nod == True:
        chop_nod_status = 'chopnod'
    if chop_nod == False: 
        chop_nod_status = 'NOchopnod'
        
    if psf_subtraction == True:
        psfsub_status = 'psfsub'
    if psf_subtraction == False:
        psfsub_status = 'NOpsfsub'
        
    if SNR_implane_plot == True:
        if psf_subtraction == True: 
            imsrcraw=plt.imshow(psf_implane, norm='log', extent=extent_array, origin='lower')
            plt.colorbar(imsrcraw, label='['+header_psf_implane['BUNIT']+']')
            plt.title('PSF Implane image '+band+' (with log norm)')
            plt.xlabel('[mas]')
            plt.ylabel('[mas]')
            plt.savefig('../outputs/'+mode+'/'+object_name+'/'+str(exptime_plot.value)+'min_'+chop_nod_status+'_'+psfsub_status+'_PSF_implane.pdf')
            plt.show()
            
        imsrcraw=plt.imshow(bgsub_implane, norm='log', extent=extent_array, origin='lower')
        plt.colorbar(imsrcraw, label='['+header_src_implane['BUNIT']+']')
        plt.title('Background substracted Implane image '+band+' (with log norm)')
        plt.xlabel('[mas]')
        plt.ylabel('[mas]')
        plt.savefig('../outputs/'+mode+'/'+object_name+'/'+str(exptime_plot.value)+'min_'+chop_nod_status+'_'+psfsub_status+'_bgsub_implane.pdf')
        plt.show()
        
        plt.imshow(snr_th_implane, extent=extent_array, origin='lower')
        plt.colorbar(label="SNR")
        if contour_plot == True:
            plt.contour(snr_th_implane, levels=[5], colors='red', linewidths=1, extent=extent_array)
            plt.contour(snr_th_implane, levels=[10], colors='orange', linewidths=1, extent=extent_array)
        plt.title("Theoretical SNR (implane)")
        plt.xlabel('[mas]')
        plt.ylabel('[mas]')
        plt.xlim(-window_lim, window_lim)
        plt.ylim(-window_lim, window_lim)
        plt.savefig('../outputs/'+mode+'/'+object_name+'/'+str(exptime_plot.value)+'min_'+chop_nod_status+'_'+psfsub_status+'_SNR_th_implane.pdf')
        plt.show()
        
        
            
            
    # TODO: change the ticks to correspond to the pixel number of the screen
    if SNR_readout_plot == True:
        if psf_subtraction == True: 
            imsrcraw=plt.imshow(psf_readout_data, norm='log', extent=extent_array, origin='lower')
            plt.colorbar(imsrcraw, label='['+header_psf_readout['BUNIT']+']')
            plt.title('PSF image (readout) (with log norm)')
            plt.xlabel('[mas]')
            plt.ylabel('[mas]')
            plt.savefig('../outputs/'+mode+'/'+object_name+'/'+str(exptime_plot.value)+'min_'+chop_nod_status+'_'+psfsub_status+'_PSF_readout.pdf')
            plt.show()
            
        imsrcraw=plt.imshow(bgsub_readout, norm='log', extent=extent_array, origin='lower')
        plt.colorbar(imsrcraw, label='['+header_src_readout['BUNIT']+']')
        plt.title('Background substracted readout image '+band+' (with log norm)')
        plt.xlabel('[mas]')
        plt.ylabel('[mas]')
        plt.savefig('../outputs/'+mode+'/'+object_name+'/'+str(exptime_plot.value)+'min_'+chop_nod_status+'_'+psfsub_status+'_bgsub_readout.pdf')
        plt.show()
        
        plt.imshow(snr_th_readout, extent=extent_array, origin='lower')
        plt.colorbar(label="SNR")
        if contour_plot == True:
            plt.contour(snr_th_readout, levels=[5], colors='red', linewidths=1,extent=extent_array)
            #plt.contour(snr_th_readout, levels=[7.5], colors='black', linewidths=1, extent=extent_array)
            plt.contour(snr_th_readout, levels=[10], colors='orange', linewidths=1, extent=extent_array)
            #plt.contour(snr_th_readout, levels=[750], colors='pink', linewidths=1, extent=extent_array)
        plt.title("Theoretical SNR (readout)")
        plt.xlabel('[mas]')
        plt.ylabel('[mas]')
        plt.xlim(-window_lim, window_lim)
        plt.ylim(-window_lim, window_lim)
        plt.savefig('../outputs/'+mode+'/'+object_name+'/'+str(exptime_plot.value)+'min_'+chop_nod_status+'_'+psfsub_status+'_SNR_th_readout.pdf')
        plt.show()
        
            
            
            
    """ FITS file CREATION"""
    if chop_nod==True:
        exptime = exptime*4
    if creating_FITs == True:
        # src
        create_fits(src_implane, header_src_implane, variable_name='src_IMPLANE', object_name=object_name, mode=mode, exptime=exptime, psfsub_status=psfsub_status, chop_nod_status=chop_nod_status)
        create_fits(src_readout_data, header_src_readout, variable_name='src_READOUT', object_name=object_name, mode=mode, exptime=exptime, psfsub_status=psfsub_status, chop_nod_status=chop_nod_status)
        
        # bgsub
        create_fits(bgsub_implane, header_src_implane, variable_name='bgsub_IMPLANE', object_name=object_name, mode=mode, exptime=exptime, psfsub_status=psfsub_status, chop_nod_status=chop_nod_status)
        create_fits(bgsub_readout, header_src_readout, variable_name='bgsub_READOUT', object_name=object_name, mode=mode, exptime=exptime, psfsub_status=psfsub_status, chop_nod_status=chop_nod_status)
        
        # snr 
        create_fits(snr_th_implane, header_src_implane,variable_name='SNR_th_IMPLANE', object_name=object_name, mode=mode, exptime=exptime, psfsub_status=psfsub_status, chop_nod_status=chop_nod_status)
        create_fits(snr_th_readout, header_src_readout, variable_name='SNR_th_READOUT', object_name=object_name, mode=mode, exptime=exptime, psfsub_status=psfsub_status, chop_nod_status=chop_nod_status)
        
        if psf_subtraction == True: 
            # psf
            create_fits(psf_implane, header_psf_implane, variable_name='PSF_IMPLANE', object_name=object_name, mode=mode, exptime=exptime, psfsub_status=psfsub_status, chop_nod_status=chop_nod_status)
            create_fits(psf_readout_data, header_src_readout, variable_name='PSF_READOUT', object_name=object_name, mode=mode, exptime=exptime, psfsub_status=psfsub_status, chop_nod_status=chop_nod_status)
        
    return [src_readout_data, src_implane, bgsub_implane, bgsub_readout, snr_th_implane, snr_th_readout]



#===========================================
#         Sensitivity_limit
#===========================================
def sensitivity_limit(snr_image, bgsub_image, chop_nod_status, SNR_limit, radius, window_lim):
    if chop_nod_status == True:
        Jy_to_ADU_factor = Jy_to_ADU_chopnod
    if chop_nod_status == False:
        Jy_to_ADU_factor = Jy_to_ADU
        
    pxl = np.where((snr_image < SNR_limit+1) & (snr_image > SNR_limit-1))

    # find out how is the pixel size in this image compared to our resolution
    bgsub_image = bgsub_image/Jy_to_ADU_factor

    mask = np.zeros_like(snr_image, dtype=bool)

    mask[pxl] = True 
    #plt.imshow(mask)
    
    bgsub_readout_masked = np.where(mask, bgsub_image, np.nan)


    """PLOT"""
    # change axis to mas instead of pxl
    ref_pt1 = np.shape(bgsub_image)[0]/2
    ref_pt2 = np.shape(bgsub_image)[1]/2
    CDELT = N_band_pixel_scale
    CDELT = CDELT.to(u.mas)
    extent_array = [-ref_pt1*CDELT.value, ref_pt1*CDELT.value, -ref_pt2*CDELT.value, ref_pt2*CDELT.value]
    center = np.shape(bgsub_image)[0]/2
    
    imsrcraw=plt.imshow(bgsub_readout_masked, extent=extent_array, origin='lower')
    plt.colorbar(imsrcraw, label='Jy')
    ax= plt.gca()
    ax.add_patch(Circle([0, 0], radius, fill=False, edgecolor='red'))
    plt.title('Brightness map with pixels where SNR ='+str(SNR_limit))
    plt.xlabel('[mas]')
    plt.ylabel('[mas]')
    plt.xlim(-window_lim, window_lim)
    plt.ylim(-window_lim, window_lim)
    plt.show()

    sum_masked = np.nansum(bgsub_readout_masked)*u.Jy
    area_1pxl = N_band_pixel_scale**2
    area_mask = np.size(pxl)*area_1pxl

    #print('Minimum flux/area with SNR = '+str(SNR_limit)+': ' + str(sum_masked/area_mask))
    brightness_lim = sum_masked/area_mask
    brightness_lim = brightness_lim.to(u.mJy/u.arcsec**2)
    print('Minimum flux/area with SNR = '+str(SNR_limit)+': ' + str(round(brightness_lim.value,2)*brightness_lim.unit))
    # find out what is the sensitivity of metis in N-band and observing mode 
    # check that the filter if is N1 or N2 (see table)
    """radius_mas = radius*N_band_pixel_scale"""
    print('Radius of the circle =', radius*u.mas)
    # for a SNR of 5 we detect surface brightness of 10 mJy/arcsec**2 (agreeing with the Table of...), radius = 180 would be the circle (middle in the grey dots) -> convert in mas. Can use this method/criterion to the others tests (flux ratio, exptime, size var) -> how do the surface brightness and radius evolve -> most important conclusion
    # double check the method even though good results (especially the one without chop nod)
    
    return brightness_lim#, radius_mas

def sensitivity_limit_5_10(snr_image, bgsub_image, chop_nod_status, radius_5, radius_10, window_lim_5, window_lim_10):
    print('======================= SNR LIMIT = 5 =======================')
    brightness_lim_5= sensitivity_limit(snr_image, bgsub_image, chop_nod_status, 5, radius_5, window_lim_5)
    print('======================= SNR LIMIT = 10 =======================')
    brightness_lim_10= sensitivity_limit(snr_image, bgsub_image, chop_nod_status, 10, radius_10, window_lim_10)
    return brightness_lim_5, brightness_lim_10#, radius_mas_5], [brightness_lim_10, radius_mas_10]




#===========================================
#           Complete function
#===========================================

def complete_Scopesim_process(total_flux, flux_ratio,
                              FWHM_polar_x, FWHM_polar_y, FWHM_disk_x, FWHM_disk_y, FWHM_unres, 
                              theta_polar, D_source, 
                              pixel_size, grid_length_factor, d_offset, 
                              object_name, mode, exptime, 
                              window_lim,
                              chop_nod=False, chop_nod_offset=3*u.arcsec,
                              psf_subtraction=False, psf_file='nothing', 
                              SNR_implane_plot=False, SNR_readout_plot=False, 
                              contour_plot=False, creating_FITs=True):
    
    tot = 7.79+1.61
    frac_polar = 7.76/tot
    frac_disk = 1.67/tot
    example_file = example_FITS
    if flux_ratio == 0: 
        print("Flux ratio value invalid")
        return 0
    
    flux_unres = total_flux*flux_ratio 
    flux_extended = total_flux-flux_unres
    flux_disk = flux_extended*frac_disk
    flux_polar = flux_extended*frac_polar
    
    
    theta_disk = theta_polar + 90*u.degree
    G = create_model_3compo(FWHM_polar_x, FWHM_polar_y, flux_polar, theta_polar,
                            FWHM_disk_x, FWHM_disk_y, flux_disk, theta_disk,
                            FWHM_unres, flux_unres, 
                            D_source, 
                            pixel_size, grid_length_factor, d_offset, 
                            example_file, 
                            mode, object_name, creating_FITs)
    
    file_src = '../outputs/'+mode+'/'+object_name+'/MODEL_3compo.fits'
    file_full_src = file_src
    SNR_results = SNR_calculation(object_name, file_src,  
                                mode, exptime, window_lim, file_full_src,
                                chop_nod, chop_nod_offset,
                                psf_subtraction, psf_file, 
                                SNR_implane_plot, SNR_readout_plot, 
                                contour_plot, creating_FITs)
    
    # SNR_results = [src_readout_data, src_implane, bgsub_implane, bgsub_readout, snr_th_implane, snr_th_readout]
    
    return G, SNR_results


