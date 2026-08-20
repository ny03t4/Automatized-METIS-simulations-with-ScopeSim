import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import matplotlib.colors as colors
from astropy.visualization import ImageNormalize, LogStretch
from astropy.io import fits
from astropy import units as u
import scopesim as sim
import os 

from variables import *

#===========================================
#                CONVERSION
#===========================================


def pc_to_mas(Dist, S):
    """Convert the size of an object in pc to its angular size in mas using the distance to the source

    Args:
        Dist (Quantity): Distance between the source and the detector [length]
        S (Quantity): Size of the object [length]

    Returns:
        Quantity: Angular size of the object in mas
    """
    Dist = Dist.to(u.pc)
    S = S.to(u.pc)
    mas = 2*np.arctan(S/2/Dist)
    mas = mas.to(u.mas)
    return mas


def mas_to_pc(theta, Dist):
    """Convert the angular size of an object in mas to its size in pc using the distance to the source

    Args:
        theta (Quantity): Angular size [angle]
        Dist (Quantity): Distance between the source and the detector [length]

    Returns:
        Quantity: Size of the object in pc
    """
    Dist = Dist.to(u.pc)
    theta = theta.to(u.radian)
    pc = 2*Dist*np.tan(theta/2)
    pc = pc.to(u.pc)
    return pc


def mean_Variance(rate, t):
    N = np.size(rate)
    return np.sum(rate)*t/N

#===========================================
#            Elliptical annulus
#===========================================

def elliptical_annulus_mean_rotated(image, a, b, CDELT, thickness, theta, window_lim):
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
    window_lim = window_lim.to(u.mas)
    
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
    plt.xlim(-window_lim.value, window_lim.value)
    plt.ylim(-window_lim.value, window_lim.value)
    plt.show()
    
    # return
    if not np.any(mask):
        return np.nan

    return image[mask].mean()

#===========================================
#                FITS FILE
#===========================================
def create_fits_model(image, pixel_size_mas, object_name, mode, model_type):
    example_data, example_header = fits.getdata(example_FITS, header=True)
    new_header= example_header.copy()

    CDELT_circinus = pixel_size_mas.to(u.degree)
    new_header['BITPIX'] = -64
    new_header['NAXIS1'] = np.shape(image)[0]
    new_header['NAXIS2'] = np.shape(image)[1]
    new_header['CRPIX1'] = np.shape(image)[0]/2
    new_header['CRPIX2'] = np.shape(image)[1]/2
    new_header["CDELT1"] = CDELT_circinus.value
    new_header["CDELT2"] = CDELT_circinus.value
    new_header['object'] = object_name
        
    file = '../models/'+mode+'/'+model_type+'_'+object_name+'.fits'
    hdu = fits.PrimaryHDU(image, header=new_header)
    hdu.writeto(file, overwrite=True)
    return 0

def create_fits_outputs(array, header, variable_name, object_name, mode, exptime, chop_nod, psf_subtraction):
    #chop & nod and PSF subtraction status
    if chop_nod == True:
        chop_nod_status = 'chopnod'
    if chop_nod == False: 
        chop_nod_status = 'NOchopnod'
        
    if psf_subtraction == True:
        psfsub_status = 'psfsub'
    if psf_subtraction == False:
        psfsub_status = 'NOpsfsub'
            
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
#                PLOTS SAVE
#===========================================
linewidth_snr = 0.5
boxanchor = (1.0,1.15)
fontsize = 12
labelsize = fontsize-2
figsize = (6,6)
    
def plot_model(image, pixel_size_mas, pixel_size_pc, object_name, mode, model_type, window_lim):
    ref_pt1 = np.shape(image)[0]/2
    ref_pt2 = np.shape(image)[1]/2
    pixel_size_mas = pixel_size_mas.to(u.mas)
    window_lim = window_lim.to(u.mas)
    
    extent_array_mas = [-ref_pt1*pixel_size_mas.value, ref_pt1*pixel_size_mas.value, -ref_pt2*pixel_size_mas.value, ref_pt2*pixel_size_mas.value]
    extent_array_pc = [-ref_pt1*pixel_size_pc.value, ref_pt1*pixel_size_pc.value, -ref_pt2*pixel_size_pc.value, ref_pt2*pixel_size_pc.value]
    
    window_lim_mas = window_lim.value
    window_lim_pxl = window_lim_mas/pixel_size_mas.value
    window_lim_pc = window_lim_pxl*pixel_size_pc.value
    
    cmap = 'turbo'
    norm = colors.PowerNorm(gamma=0.5)
    origin = 'lower'
    file_save_format = '../models/'+mode+'/'+model_type+'/'+object_name
    colorbar_label = '[Jy]'
    
    # labels in pxl - FULL IMAGE
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, origin=origin, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, size=fontsize)
    cbar.ax.tick_params(labelsize=labelsize) 
    ax.set_xlabel('[pixel]', fontsize=fontsize)
    ax.set_ylabel('[pixel]', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    plt.savefig(file_save_format+'_pxl_FULL IMAGE.pdf')
    plt.show()
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, origin=origin, norm=norm, cmap='turbo')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, size=fontsize)
    cbar.ax.tick_params(labelsize=labelsize) 
    ax.set_xlabel('[pixel]', fontsize=fontsize)
    ax.set_ylabel('[pixel]', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    plt.savefig(file_save_format+'_pxl_normlog_FULL IMAGE.pdf')
    plt.show()
        
    # labels in pxl
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, origin=origin, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, size=fontsize)
    cbar.ax.tick_params(labelsize=labelsize) 
    ax.set_xlabel('[pixel]', fontsize=fontsize)
    ax.set_ylabel('[pixel]', fontsize=fontsize)
    ax.set_xlim(int(ref_pt1)-window_lim_pxl, int(ref_pt1)+window_lim_pxl)
    ax.set_ylim(int(ref_pt2)-window_lim_pxl, int(ref_pt2)+window_lim_pxl)
    plt.savefig(file_save_format+'_pxl.pdf')
    plt.show()
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, origin=origin,  norm=norm, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, size=fontsize)
    cbar.ax.tick_params(labelsize=labelsize) 
    ax.set_xlabel('[pixel]', fontsize=fontsize)
    ax.set_ylabel('[pixel]', fontsize=fontsize)
    ax.set_xlim(int(ref_pt1)-window_lim_pxl, int(ref_pt1)+window_lim_pxl)
    ax.set_ylim(int(ref_pt2)-window_lim_pxl, int(ref_pt2)+window_lim_pxl)
    plt.savefig(file_save_format+'_pxl_normlog.pdf')
    plt.show()
    
    # labels in mas
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, origin=origin, cmap=cmap, extent=extent_array_mas)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, size=fontsize)
    cbar.ax.tick_params(labelsize=labelsize) 
    ax.set_xlabel('[mas]', fontsize=fontsize)
    ax.set_ylabel('[mas]', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.set_xlim(-window_lim_mas, window_lim_mas)
    ax.set_ylim(-window_lim_mas, window_lim_mas)
    plt.savefig(file_save_format+'_mas.pdf')
    plt.show()
        
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, origin=origin, cmap=cmap, extent=extent_array_mas,  norm=norm)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, size=fontsize)
    cbar.ax.tick_params(labelsize=labelsize) 
    ax.set_xlabel('[mas]', fontsize=fontsize)
    ax.set_ylabel('[mas]', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.set_xlim(-window_lim_mas, window_lim_mas)
    ax.set_ylim(-window_lim_mas, window_lim_mas)
    plt.savefig(file_save_format+'_mas_normlog.pdf')
    plt.show()
    
    # labels in pc
    if pixel_size_pc.value != 0:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(image, origin=origin, cmap=cmap, extent=extent_array_pc)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label, size=fontsize)
        cbar.ax.tick_params(labelsize=labelsize) 
        ax.set_xlabel('[pc]', fontsize=fontsize)
        ax.set_ylabel('[pc]', fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=labelsize)
        ax.set_xlim(-window_lim_pc, window_lim_pc)
        ax.set_ylim(-window_lim_pc, window_lim_pc)
        plt.savefig(file_save_format+'_pc.pdf')
        plt.show()
            
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(image, origin=origin, cmap=cmap, extent=extent_array_pc,  norm=norm)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label, size=fontsize)
        cbar.ax.tick_params(labelsize=labelsize) 
        ax.set_xlabel('[pc]', fontsize=fontsize)
        ax.set_ylabel('[pc]', fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=labelsize)
        ax.set_xlim(-window_lim_pc, window_lim_pc)
        ax.set_ylim(-window_lim_pc, window_lim_pc)
        plt.savefig(file_save_format+'_pc_normlog.pdf')
        plt.show() 
    return 0

def plot_outputs(image, variable_name, CDELT, unit, exptime, chop_nod, psf_subtraction, object_name, mode, window_lim, contour_plot, norm):
    print('===========================================')
    print('                PLOT '+variable_name)
    print('===========================================')
    # change axis to mas instead of pxl
    ref_pt1 = np.shape(image)[0]/2
    ref_pt2 = np.shape(image)[1]/2
    CDELT = CDELT.to(u.mas)
    extent_array = [-ref_pt1*CDELT.value, ref_pt1*CDELT.value, -ref_pt2*CDELT.value, ref_pt2*CDELT.value]
    exptime_plot = exptime.to(u.minute)
    
    #chop & nod and PSF subtraction status
    if chop_nod == True:
        chop_nod_status = 'chopnod'
    if chop_nod == False: 
        chop_nod_status = 'NOchopnod'
        
    if psf_subtraction == True:
        psfsub_status = 'psfsub'
    if psf_subtraction == False:
        psfsub_status = 'NOpsfsub'
        
    # param plot
    cmap = 'turbo'
        
    if norm == 'log':
        norm = 'log'
    if norm == 'power':
        norm = colors.PowerNorm(gamma=0.5)
    if norm == "ds9":
        norm = ImageNormalize(vmin=np.nanmin(image), vmax=np.nanmax(image),stretch=LogStretch(1000))
        
    if variable_name == 'SKY_IMPLANE' or variable_name == 'SKY_READOUT' or (variable_name == 'SRC_READOUT' and chop_nod==True):
        cmap = 'viridis'
        norm = 'log'
            
    origin = 'lower'
    colorbar_label = '['+unit+']'
    file_save_format = '../outputs/'+mode+'/'+object_name+'/'+str(exptime_plot.value)+'min_'+chop_nod_status+'_'+psfsub_status+'_'+variable_name
    
    colorsnr5 = '#E50000'
    colorsnr10 = '#9A0EEA'
    
    # FULL IMAGE plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, extent=extent_array, origin=origin, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, size=fontsize)
    cbar.ax.tick_params(labelsize=labelsize) 
    if contour_plot == True:
        contour = ax.contour(image, levels=[5, 10], colors=(colorsnr5, colorsnr10), linewidths=linewidth_snr, extent=extent_array)
        handles, labels = contour.legend_elements()
        ax.legend(handles, ['SNR = 5', 'SNR = 10'], loc='upper right', bbox_to_anchor=boxanchor, fontsize=labelsize)
    ax.set_xlabel('[mas]', fontsize=fontsize)
    ax.set_ylabel('[mas]', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    plt.savefig(file_save_format+'_FULL IMAGE.pdf')
    plt.show()
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, extent=extent_array, origin=origin, cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, size=fontsize)
    cbar.ax.tick_params(labelsize=labelsize) 
    if contour_plot == True:
        contour = ax.contour(image, levels=[5, 10], colors=(colorsnr5, colorsnr10), linewidths=linewidth_snr, extent=extent_array)
        handles, labels = contour.legend_elements()
        ax.legend(handles, ['SNR = 5', 'SNR = 10'], loc='upper right', bbox_to_anchor=boxanchor, fontsize=labelsize)
    ax.set_xlabel('[mas]', fontsize=fontsize)
    ax.set_ylabel('[mas]', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    plt.savefig(file_save_format+'_FULL IMAGE_normlog.pdf')
    plt.show()
    
    # Zoomed image plot 
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, extent=extent_array, origin=origin, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, size=fontsize)
    cbar.ax.tick_params(labelsize=labelsize) 
    if contour_plot == True:
        contour = ax.contour(image, levels=[5, 10], colors=(colorsnr5, colorsnr10), linewidths=linewidth_snr, extent=extent_array)
        handles, labels = contour.legend_elements()
        ax.legend(handles, ['SNR = 5', 'SNR = 10'], loc='upper right', bbox_to_anchor=boxanchor, fontsize=labelsize)
    ax.set_xlabel('[mas]', fontsize=fontsize)
    ax.set_ylabel('[mas]', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.set_xlim(-window_lim, window_lim)
    ax.set_ylim(-window_lim, window_lim)
    plt.savefig(file_save_format+'.pdf')
    plt.show()
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, extent=extent_array, origin=origin, cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, size=fontsize)
    cbar.ax.tick_params(labelsize=labelsize) 
    if contour_plot == True:
        contour = ax.contour(image, levels=[5, 10], colors=(colorsnr5, colorsnr10), linewidths=linewidth_snr, extent=extent_array)
        handles, labels = contour.legend_elements()
        ax.legend(handles, ['SNR = 5', 'SNR = 10'], loc='upper right', bbox_to_anchor=boxanchor, fontsize=labelsize)
    ax.set_xlabel('[mas]', fontsize=fontsize)
    ax.set_ylabel('[mas]', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.set_xlim(-window_lim, window_lim)
    ax.set_ylim(-window_lim, window_lim)
    plt.savefig(file_save_format+'_normlog.pdf')
    plt.show()
    return 0


def plot_mask(image, radius_mas, variable_name, CDELT, unit, exptime, chop_nod, psf_subtraction, object_name, mode, window_lim):
    print('===========================================')
    print('                PLOT '+variable_name)
    print('===========================================')
    # change axis to mas instead of pxl
    ref_pt1 = np.shape(image)[0]/2
    ref_pt2 = np.shape(image)[1]/2
    CDELT = CDELT.to(u.mas)
    extent_array = [-ref_pt1*CDELT.value, ref_pt1*CDELT.value, -ref_pt2*CDELT.value, ref_pt2*CDELT.value]
    exptime_plot = exptime.to(u.minute)
    #radius_mas = radius_pxl*CDELT.value
    
    #chop & nod and PSF subtraction status
    if chop_nod == True:
        chop_nod_status = 'chopnod'
    if chop_nod == False: 
        chop_nod_status = 'NOchopnod'
        
    if psf_subtraction == True:
        psfsub_status = 'psfsub'
    if psf_subtraction == False:
        psfsub_status = 'NOpsfsub'
        
    # param plot
    cmap = 'turbo'
        
    origin = 'lower'
    colorbar_label = '['+unit+']'
    file_save_format = '../outputs/'+mode+'/'+object_name+'/'+str(exptime_plot.value)+'min_'+chop_nod_status+'_'+psfsub_status+'_'+variable_name
    
    if variable_name == 'SNR = 5':
        color_circle = '#E50000'
    if variable_name == 'SNR = 10':
        color_circle = '#9A0EEA'
        
    # FULL IMAGE plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, extent=extent_array, origin=origin, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, size=fontsize)
    cbar.ax.tick_params(labelsize=labelsize)
    circle = ax.add_patch(Circle([0, 0], radius_mas, fill=False, edgecolor=color_circle))
    circle.set_label('Circle of highest point density, '+variable_name)
    ax.legend(loc='upper right', bbox_to_anchor=boxanchor, fontsize=labelsize)
    ax.set_xlabel('[mas]', fontsize=fontsize)
    ax.set_ylabel('[mas]', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    plt.savefig(file_save_format+'_FULL IMAGE.pdf')
    plt.show()       
    
    # Zoomed image plot 
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, extent=extent_array, origin=origin, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, size=fontsize)
    cbar.ax.tick_params(labelsize=labelsize)
    circle = ax.add_patch(Circle([0, 0], radius_mas, fill=False, edgecolor=color_circle))
    circle.set_label('Circle of highest point density, '+variable_name)
    ax.legend(loc='upper right', bbox_to_anchor=boxanchor, fontsize=labelsize)
    ax.set_xlabel('[mas]', fontsize=fontsize)
    ax.set_ylabel('[mas]', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.set_xlim(-window_lim, window_lim)
    ax.set_ylim(-window_lim, window_lim)
    plt.savefig(file_save_format+'.pdf')
    plt.show()       
    return 0

#===========================================
#            UNIFORM DISK
#===========================================

def Uniform_disk_profile(flux_per_pixel, radius, pixel_size_mas, mode, object_name, creating_plot_and_FITS, window_lim):
    """Create a 2D array containing a uniform filled disk.

    Args:
        

    Returns:
        Array(float): 2D array containing a uniform filled disk.
    """
    model_type = 'Uniform_disk'
    pixel_size_mas = pixel_size_mas.to(u.mas)
    flux_per_pixel = flux_per_pixel.to(u.Jy)
    
    if mode == 'img_n':
        grid_length = int(METIS_N_band_pixel_scale.value/pixel_size_mas.value)*screen_size
    if mode == 'img_lm':
        grid_length = int(METIS_L_band_pixel_scale.value/pixel_size_mas.value)*screen_size
    
    if grid_length % 2 == 0:
        grid_length += 1
    Grid_size = (grid_length, grid_length)
    array = np.zeros(Grid_size)
    cx = grid_length/2
    cy =  grid_length/2
    
    y, x = np.ogrid[:Grid_size[0], :Grid_size[1]]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2

    array[mask] = flux_per_pixel
    
    # Creation of a FITS file to store the Gaussian
    if creating_plot_and_FITS == True:
        create_fits_model(array, pixel_size_mas, object_name, mode, model_type)
        plot_model(array, pixel_size_mas, 0*u.pc, object_name, mode, model_type, window_lim)
    
    return array


#===========================================
#            CIRCULAR GAUSSIAN
#===========================================

def Circular_Gaussian_profile(FWHM, flux, pixel_size_mas, Dist_source, mode, object_name, creating_plot_and_FITS, window_lim):
    """Generate a 2D circular Gaussian profile and create a FITS file.

    Args:
        FWHM (Quantity): FWHM of the source
        flux (Quantity): Total flux of the source received by the detector
        pixel_size_mas (Quantity): Size of the pixel in pc

    Returns:
        Array(float): 2D array containing the 2D circular Gaussian profile
    """
    model_type = 'Circular_Gaussian'
    FWHM = FWHM.to(u.pc)
    FWHM = FWHM.value
    pixel_size_mas = pixel_size_mas.to(u.mas)
    pixel_size_pc = mas_to_pc(pixel_size_mas, Dist_source)
    flux = flux.to(u.Jy)
    
    sigma = FWHM/(2* np.sqrt(2*np.log(2)))/pixel_size_pc.value
    if mode == 'img_n':
        grid_length = int(METIS_N_band_pixel_scale.value/pixel_size_mas.value)*screen_size
    if mode == 'img_lm':
        grid_length = int(METIS_L_band_pixel_scale.value/pixel_size_mas.value)*screen_size
    
    if grid_length % 2 == 0:
        grid_length += 1
    Grid_size = (grid_length, grid_length)

    x = [i for i in range(Grid_size[0])]
    y = [i for i in range(Grid_size[1])]
    # Generation limit (to reduce computation time)
    gen_lim = gen_fact*sigma
    
    # The source is centered at the center of the grid
    x_c = int(Grid_size[0]/2) 
    y_c = int(Grid_size[1]/2)
    I = 1
    Gauss = np.zeros((len(x), len(y)))
    if flux.value != 0:
        for i in range(len(x)):
            for j in range(len(y)):
                if abs(x[i]-x_c) <= gen_lim and abs(y[j]-y_c) <= gen_lim:
                    Gauss[i,j] = I*np.exp(-0.5*(((x[i]-x_c)/sigma)**2 + ((y[j]-y_c)/sigma)**2))
        
        # To normalize the gaussian following the value of the total flux of the source
        somme = np.sum(Gauss)
        ratio = flux.value/somme 
        Gauss *= ratio
    
    # Creation of a FITS file to store the Gaussian
    if creating_plot_and_FITS == True:
        create_fits_model(Gauss, pixel_size_mas, object_name, mode, model_type)
        plot_model(Gauss, pixel_size_mas, pixel_size_pc, object_name, mode, model_type, window_lim)
    
    return Gauss


#===========================================
#            ELLIPTICAL GAUSSIAN
#===========================================

def Elliptical_Gaussian_profile(FWHM_x, FWHM_y, theta, flux, pixel_size_mas, Dist_source, mode, object_name, creating_plot_and_FITS, window_lim):
    """Generate a 2D Elliptical Gaussian profile.

    Args:
        FWHM_x (Quantity): FWHM of the source in x axis
        FWHM_y (Quantity): FWHM of the source in y axis
        flux (Quantity): Total flux of the source received by the detector
        pixel_size_mas (Quantity): Size of the pixel in pc

    Returns:
        Array(float): 2D array containing the 2D elliptical Gaussian profile
    """
    model_type = 'Elliptical_Gaussian'
    FWHM_x = FWHM_x.to(u.pc)
    FWHM_x = FWHM_x.value
    FWHM_y = FWHM_y.to(u.pc)
    FWHM_y = FWHM_y.value
    theta = theta.to(u.radian)
    theta = theta.value 
    
    pixel_size_mas = pixel_size_mas.to(u.mas)
    pixel_size_pc = mas_to_pc(pixel_size_mas, Dist_source)
    flux = flux.to(u.Jy)
    
    sigma_x = FWHM_x/(2* np.sqrt(2*np.log(2)))/pixel_size_pc.value
    sigma_y = FWHM_y/(2* np.sqrt(2*np.log(2)))/pixel_size_pc.value
    #grid_length = int(grid_length_factor*np.max([sigma_x, sigma_y]))
    if mode == 'img_n':
        grid_length = int(METIS_N_band_pixel_scale.value/pixel_size_mas.value)*screen_size
    if mode == 'img_lm':
        grid_length = int(METIS_L_band_pixel_scale.value/pixel_size_mas.value)*screen_size
            
    if grid_length % 2 == 0:
        grid_length += 1
    Grid_size = (grid_length, grid_length)

    x = [i for i in range(Grid_size[0])]
    y = [i for i in range(Grid_size[1])]
    # Generation limit (to reduce computation time)
    gen_lim = max(gen_fact*sigma_x, gen_fact*sigma_y)
            
    # The source is centered at the center of the grid
    x_c = int(Grid_size[0]/2) 
    y_c = int(Grid_size[1]/2)
    I = 1
    Gauss = np.zeros((len(x), len(y)))
    a = np.cos(theta)**2/(2*sigma_x**2)+np.sin(theta)**2/(2*sigma_y**2)
    c = np.cos(theta)**2/(2*sigma_y**2)+np.sin(theta)**2/(2*sigma_x**2) 
    b = -np.cos(theta)*np.sin(theta)/(2*sigma_x**2)+np.cos(theta)*np.sin(theta)/(2*sigma_y**2)
    
    if flux.value != 0:
        for i in range(len(x)):
            for j in range(len(y)):
                if abs(x[i]-x_c) <= gen_lim and abs(y[j]-y_c) <= gen_lim:
                    Gauss[i,j] = I*np.exp(-(a*(x[i]-x_c)**2 + 2*b*(x[i]-x_c)*(y[j]-y_c) + c*(y[j]-y_c)**2))
        
        # To normalize the gaussian following the value of the total flux of the source
        somme = np.sum(Gauss)
        ratio = flux.value/somme 
        Gauss *= ratio

    # Creation of a FITS file to store the Gaussian
    if creating_plot_and_FITS == True:
        create_fits_model(Gauss, pixel_size_mas, object_name, mode, model_type)
        plot_model(Gauss, pixel_size_mas, pixel_size_pc, object_name, mode, model_type, window_lim)
        
    return Gauss

#===========================================
#        1 pt source + ELLIPTICAL GAUSSIAN
#===========================================
def create_model_3compo(FWHM_polar_x, FWHM_polar_y, flux_polar, theta_polar,
                        FWHM_disk_x, FWHM_disk_y, flux_disk, theta_disk,
                        FWHM_unres, flux_unres, 
                        Dist_source, 
                        pixel_size_mas, d_offset, 
                        mode, object_name, creating_plot_and_FITS, window_lim):
    
    if flux_disk.value == 0 or flux_polar.value == 0 or flux_unres.value == 0:
        model_type = "2_Gaussians"
    else:
        model_type = '3_Gaussians'
        
    pixel_size_mas = pixel_size_mas.to(u.mas)
    pixel_size_pc = mas_to_pc(pixel_size_mas, Dist_source)
    
    G_unres = Circular_Gaussian_profile(FWHM_unres, 
                                        flux_unres, 
                                        pixel_size_mas, Dist_source, mode, object_name, False, window_lim)
    
    G_polar = Elliptical_Gaussian_profile(FWHM_polar_x, FWHM_polar_y, 
                                          theta_polar, flux_polar, 
                                          pixel_size_mas, Dist_source, mode, object_name, False, window_lim)
    
    G_disk = Elliptical_Gaussian_profile(FWHM_disk_x, FWHM_disk_y, 
                                         theta_disk, flux_disk, 
                                         pixel_size_mas, Dist_source, mode, object_name, False, window_lim)
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
    if creating_plot_and_FITS == True:
        create_fits_model(G_polar, pixel_size_mas, object_name, mode, model_type)
        plot_model(G_polar, pixel_size_mas, pixel_size_pc, object_name, mode, model_type, window_lim)
    
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
        metis.observe(image, update=True)

    implane_data = metis.image_planes[0].data #*u.photon/u.s # [ph/s]
    header_implane = metis.image_planes[0].header
    
    if chop_nod == True and file != 'sky':
        chop_nod_offset = chop_nod_offset.to(u.arcsec)
        metis['chop_nod'].include = True
        metis["chop_nod"].meta["chop_offsets"] = [chop_nod_offset.value, 0]
        metis["chop_nod"].meta["nod_offsets"] = [0, chop_nod_offset.value]
        print("Chopping:", np.array([chop_nod_offset.value, 0]))
        print("Nodding: ", np.array([0, chop_nod_offset.value]))
        
    readout  = metis.readout(dit=dit.value, ndit=ndit)[0] 
    readout_data = readout[1].data #*u.adu # [ADU]
    header_readout = readout[1].header
    print(metis.cmds['!DET.mode'])
    return implane_data, readout, readout_data, header_implane, header_readout

#===========================================
#                CHOP NOD
#===========================================


def chop_nod_fct(metis_readout, chop_nod_offset, dit, ndit, mode):
    image = metis_readout[1].data
    metis_readout[0].header['DIT'] = dit.value
    metis_readout[0].header['NDIT'] = ndit*4 
    metis_readout[0].header['EXPTIME'] = dit.value * ndit*4
    
    if mode == "img_n":
        pixel_scale = METIS_N_band_pixel_scale
    if mode == "img_lm":
        pixel_scale = METIS_L_band_pixel_scale
        
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

def SNR_TH(bgsub_psfsub_image, bgsub_psf, bg, n_pix, ron, darkCurrent, dit, ndit, chop_nod):
    """Calculation SNR following theoretical formula

    Args:
        bgsub_psfsub_signal (array): background and psf subtracted image [e-] !!! MUTLIPLIED BY THE GAIN (readout image) OR QE and t=dit*ndit (implane image)
        bgsub_signal (array): background subtracted image [e-] !!! MUTLIPLIED BY THE GAIN (readout image) OR QE and t=dit*ndit (implane image)
        bgsub_psf (array): background subtracted psf image [e-] !!! MUTLIPLIED BY THE GAIN (readout image) OR QE and t=dit*ndit (implane image)
        bg (array): background image [e-] !!! MUTLIPLIED BY THE GAIN (readout image) OR QE and t=dit*ndit (implane image)
        n_pix (int): number of pixels on the screen
        ron (float): Read-out-noise
        darkCurrent (float): Darkcurrent
        dit (float): dit
        ndit (float): ndit  !!! *4 in case of chop_nod == True

    Returns:
        array: SNR map
    """
    
    """ron_per_pixel = ron/n_pix  # [e-/integration^(1/2)/pxl]
    darkCurrent_per_pixel = darkCurrent/n_pix # [e-/s/pxl]
    t = dit*ndit # [s]
    snr_th_map = t * bgsub_psfsub_image / np.sqrt(t*bgsub_image + t*bgsub_psf + bg*t + t*darkCurrent_per_pixel + ndit*ron_per_pixel**2)"""
    
    ron_per_pixel = ron/n_pix # [e-/pxl]
    darkCurrent_per_pixel = darkCurrent/n_pix # [e-/s/pxl]
    t = dit.value*ndit  # [s]
    
    """if np.sum(bgsub_psf) != 0:
        sigma2_psf =  bgsub_psf*t + bg*t + t*darkCurrent_per_pixel + ndit*ron_per_pixel**2
    if np.sum(bgsub_psf) == 0:
        sigma2_psf = 0"""
    
    if chop_nod == False:
        snr_th_map =  bgsub_psfsub_image*t / np.sqrt(bgsub_psfsub_image*t + bgsub_psf*t + bg*t + t*darkCurrent_per_pixel + ndit*ron_per_pixel**2)
        
        
    if chop_nod == True: 
        snr_th_map =  bgsub_psfsub_image/4*t / np.sqrt(bgsub_psfsub_image/4*t + bgsub_psf*t + bg*t + t*darkCurrent_per_pixel + ndit*ron_per_pixel**2)
    
    return snr_th_map 

#===========================================
#             FULL SNR PROCESS
#===========================================

def SNR_calculation(object_name, file_src, filter,
                    mode, exptime, dit, window_lim,
                    chop_nod=False, chop_nod_offset=3*u.arcsec,
                    psf_subtraction=False, psf_file='nothing', 
                    IMPLANE_plot=False, READOUT_plot=False, 
                    contour_plot=False, norm='ds9',
                    creating_plot_and_FITS=True): 
    
      
    # METIS set up
    exptime = exptime.to(u.s)
    dit = dit.to(u.s)
    """if dit.value < 0.011:
        print('ERROR: DIT value lower than MIN DIT.')
        return 0"""
    
    ndit = int(exptime/dit)
    
    cmd = sim.UserCommands(use_instrument="METIS", set_modes=[mode], properties={"!OBS.filter_name": filter, "!OBS.exptime": exptime.value, "!OBS.dit": dit.value, "!OBS.ndit": ndit})
    metis = sim.OpticalTrain(cmd)
    
    # Sky observation
    print('======================= SKY OBSERVATION =======================')
    print('DIT ='+str(dit)+', NDIT = '+str(ndit)+', exptime = '+str(dit*ndit)+', filter = '+cmd['!OBS.filter_name'])
    sky_implane, sky_readout, sky_readout_data, header_sky_implane, header_sky_readout = METIS_observation('sky', chop_nod_offset, metis, dit, ndit, chop_nod)
    """metis.observe()      # blank sky
    
    sky_implane = metis.image_planes[0].data # [ph/s]
    sky_readout = metis.readout(dit=dit, ndit=ndit)[0]
    sky_readout_data = sky_readout[1].data
    dit, ndit = metis.cmds["!OBS.dit"], metis.cmds["!OBS.ndit"]"""
    
    
            
    # PSF subtraction
    if psf_subtraction == True:
        print('======================= PSF OBSERVATION =======================')
        print('DIT ='+str(dit)+', NDIT = '+str(ndit)+', exptime = '+str(dit*ndit))
        psf_implane, psf_readout, psf_readout_data, header_psf_implane, header_psf_readout = METIS_observation(psf_file, chop_nod_offset, metis, dit, ndit, False)
        """if chop_nod == True:
            psf_chop_nod = chop_nod_fct(psf_readout, chop_nod_offset, dit, ndit, mode)"""
            
    # Object observation
    print('======================= OBJECT OBSERVATION =======================')
    if chop_nod == True:
        ndit = ndit/4
    print('DIT ='+str(dit)+', NDIT = '+str(ndit)+', exptime = '+str(dit*ndit))
    src_implane, src_readout, src_readout_data, header_src_implane, header_src_readout  = METIS_observation(file_src, chop_nod_offset, metis, dit, ndit, chop_nod)
    if chop_nod == True:
        bgsub_readout = chop_nod_fct(src_readout, chop_nod_offset, dit, ndit, mode) 
        # Sky & psf trimming (because chop & nod reduces the size of the image)    
        length = np.shape(bgsub_readout)[0]
        half_length = int(length/2)
        sky_readout_data = sky_readout_data[screen_center-half_length :screen_center+half_length,
                                            screen_center-half_length :screen_center+half_length]
        if psf_subtraction == True:
            psf_readout_data = psf_readout_data[screen_center-half_length :screen_center+half_length,
                                            screen_center-half_length :screen_center+half_length]
        
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
        bgsub_psf_implane = psf_implane - sky_implane
        
        bgsub_psf_readout = psf_readout_data - sky_readout_data
        bgsub_readout = bgsub_readout - bgsub_psf_readout
        
    """# Full source observation 
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
            bgsub_readout_full_src = chop_nod_fct(full_src_readout, chop_nod_offset, dit, ndit, mode) 
        if chop_nod == False:
            bgsub_readout_full_src = full_src_readout_data - sky_readout_data"""
           
    

    """SNR CALCULATION"""
    # Detector information 
    n_pix_implane = np.shape(bgsub_implane)[0]**2
    n_pix_readout = np.shape(bgsub_readout)[0]**2
    #metis.cmds["!DET"]
    ron = metis.cmds["!DET.readout_noise"] #*u.electron**(1/2) # [e- rms]
    darkCurrent = metis.cmds['!DET.dark_current'] #*u.electron/u.s # [e-/s]
    gain = metis.cmds["!DET.gain"] #*u.electron/u.adu# [e-/ADU]
    
    if mode == "img_n":
        QE = 0.8 #*u.electron/u.photon # [-]
    if mode == "img_lm":
        QE =  0.8574456 #*u.electron/u.photon# [-]
        
    if chop_nod == True:
        ndit = ndit*4
    print('FOR SNR CALCULATION: DIT ='+str(dit)+', NDIT = '+str(ndit)+', exptime = '+str(dit*ndit))
    # Theoretical SNR map (implane)
    snr_th_implane = SNR_TH(bgsub_implane*QE, bgsub_psf_implane*QE, sky_implane*QE, n_pix_implane, ron, darkCurrent, dit, ndit, chop_nod)
    
    # Theoretical SNR map (readout)
    snr_th_readout = SNR_TH(bgsub_readout/dit.value*gain, bgsub_psf_readout/dit.value*gain, sky_readout_data/dit.value*gain, n_pix_readout, ron, darkCurrent, dit, ndit, chop_nod)
    
    # Estimation of SNR from the data
    # snr_emp_map, snr_emp_mean, snr_th_cropped = SNR_DATA(bgsub_readout*gain, sky_readout_data*gain, dit, ndit, subarray_size, distance_noise_subarray, chop_nod)
    
    
    """" PLOTS """
    CDELT1 = header_src_readout['CDELT1'] *u.arcsec

    if IMPLANE_plot == True:
        if psf_subtraction == True: 
            plot_outputs(psf_implane, 'PSF_IMPLANE', CDELT1, header_psf_implane['BUNIT'],
                         exptime, chop_nod, psf_subtraction, object_name, mode,
                         window_lim, contour_plot=False, norm=norm)
           
        plot_outputs(sky_implane, 'SKY_IMPLANE', CDELT1, header_sky_implane['BUNIT'],
                                 exptime, chop_nod, psf_subtraction, object_name, mode,
                                 window_lim, contour_plot=False, norm=norm)
        
        plot_outputs(src_implane, 'SRC_IMPLANE', CDELT1, header_src_implane['BUNIT'],
                     exptime, chop_nod, psf_subtraction, object_name, mode,
                     window_lim, contour_plot=False, norm=norm)
        
        plot_outputs(bgsub_implane, 'BGSUB_IMPLANE', CDELT1, header_src_implane['BUNIT'],
                     exptime, chop_nod, psf_subtraction, object_name, mode,
                     window_lim, contour_plot=False, norm=norm)
        
        plot_outputs(snr_th_implane, 'SNR_th_IMPLANE', CDELT1, '-',
                             exptime, chop_nod, psf_subtraction, object_name, mode,
                             window_lim, contour_plot, norm=norm)
        
    if READOUT_plot == True:
        if psf_subtraction == True: 
            plot_outputs(psf_readout_data, 'PSF_READOUT', CDELT1, header_psf_readout['BUNIT'],
                                     exptime, chop_nod, psf_subtraction, object_name, mode,
                                     window_lim, contour_plot=False, norm=norm)
            
        plot_outputs(sky_readout_data, 'SKY_READOUT', CDELT1, header_sky_readout['BUNIT'],
                        exptime, chop_nod, psf_subtraction, object_name, mode,
                        window_lim, contour_plot=False, norm=norm)
        
        plot_outputs(src_readout_data, 'SRC_READOUT', CDELT1, header_src_readout['BUNIT'],
                             exptime, chop_nod, psf_subtraction, object_name, mode,
                             window_lim, contour_plot=False, norm=norm)
        
        plot_outputs(bgsub_readout, 'BGSUB_READOUT', CDELT1, header_src_readout['BUNIT'],
                        exptime, chop_nod, psf_subtraction, object_name, mode,
                        window_lim, contour_plot=False, norm=norm)
        
        plot_outputs(snr_th_readout, 'SNR_th_READOUT', CDELT1, '-',
                                exptime, chop_nod, psf_subtraction, object_name, mode,
                                window_lim, contour_plot, norm=norm)
        
            
    """ FITS file CREATION"""
    if creating_plot_and_FITS == True:
        # src
        create_fits_outputs(src_implane, header_src_implane, 'src_IMPLANE', object_name, mode, exptime, chop_nod, psf_subtraction)
        create_fits_outputs(src_readout_data, header_src_readout, 'src_READOUT', object_name, mode, exptime, chop_nod, psf_subtraction)
        
        # bgsub
        create_fits_outputs(bgsub_implane, header_src_implane, 'bgsub_IMPLANE', object_name, mode, exptime, chop_nod, psf_subtraction)
        create_fits_outputs(bgsub_readout, header_src_readout, 'bgsub_READOUT', object_name, mode, exptime, chop_nod, psf_subtraction)
        
        # snr 
        create_fits_outputs(snr_th_implane, header_src_implane, 'SNR_th_IMPLANE', object_name, mode, exptime, chop_nod, psf_subtraction)
        create_fits_outputs(snr_th_readout, header_src_readout, 'SNR_th_READOUT', object_name, mode, exptime, chop_nod, psf_subtraction)
        
        if psf_subtraction == True: 
            # psf
            create_fits_outputs(psf_implane, header_psf_implane, 'PSF_IMPLANE', object_name, mode, exptime, chop_nod, psf_subtraction)
            create_fits_outputs(psf_readout_data, header_src_readout, 'PSF_READOUT', object_name, mode, exptime, chop_nod, psf_subtraction)
        
    return [src_readout_data, src_implane, bgsub_implane, bgsub_readout, snr_th_implane, snr_th_readout]



#===========================================
#         Sensitivity_limit
#===========================================


"""def distance_to_closest_and_farthest_pixel(array):
    # Find coordinates of non-null pixels
    indices = np.argwhere(array != np.nan)
    
    # Center of the array
    cy = (array.shape[0]) / 2
    cx = (array.shape[1]) / 2
    
    # Distance of every non-null pixel to the center
    distances = [np.sqrt((indices[i,0] - cy)**2 +(indices[i,1] - cx)**2) for i in range(indices.shape[0])]

    # Find closest pixel
    i = np.argmin(distances)
    j = np.argmax(distances)
    return distances[i], distances[j]"""
    
def find_radius(image):
    # Example: image is your 2D array
    ny, nx = image.shape

    # Center of the array
    cy = ny/ 2
    cx = nx / 2

    # Coordinates of non-null pixels
    y, x = np.where(~np.isnan(image))

    # Distance from center
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_max = np.max(r)
    dr = 1.0  # radial bin width in pixels

    bins = np.arange(0, r_max + dr, dr)

    # Number of points in each annulus
    counts, edges = np.histogram(r, bins=bins)

    # Area of each annulus
    areas = np.pi * (edges[1:]**2 - edges[:-1]**2)

    # Point density
    density = counts / areas

    # Radius corresponding to each density value
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    
    r_peak = r_centers[np.argmax(density)]

    #print("Radius of maximum density:", r_peak)
    return r_peak



def sensitivity_limit(snr_image, bgsub_image, SNR_limit, Jy_to_ADU_factor, exptime, chop_nod, psf_subtraction, object_name, window_lim, mode):
        
    pxl = np.where((snr_image < SNR_limit+0.5) & (snr_image > SNR_limit-0.5))
    # find out how is the pixel size in this image compared to our resolution
    bgsub_image = bgsub_image/Jy_to_ADU_factor

    mask = np.zeros_like(snr_image, dtype=bool)

    mask[pxl] = True 
    #plt.imshow(mask)
    
    bgsub_readout_masked = np.where(mask, bgsub_image, np.nan)
    #min_dist, max_dist = distance_to_closest_and_farthest_pixel(bgsub_readout_masked)
    radius_pxl = find_radius(bgsub_readout_masked)
    radius_mas = radius_pxl*METIS_N_band_pixel_scale

    """PLOT"""
    plot_mask(bgsub_readout_masked*1000, radius_mas.value, 'SNR = '+str(SNR_limit), METIS_N_band_pixel_scale, 'mJy', exptime, chop_nod, psf_subtraction, object_name, mode, window_lim)
    
    
    # Brightness determination
    sum_masked = np.nansum(bgsub_readout_masked)*u.Jy
    area_1pxl = METIS_N_band_pixel_scale**2
    area_mask = np.size(pxl)*area_1pxl

    #print('Minimum flux/area with SNR = '+str(SNR_limit)+': ' + str(sum_masked/area_mask))
    brightness_lim = sum_masked/area_mask
    brightness_lim = brightness_lim.to(u.mJy/u.arcsec**2)
    print('Minimum flux/area with SNR'+str(SNR_limit)+': ' + str(round(brightness_lim.value,2)*brightness_lim.unit))
    # find out what is the sensitivity of metis in N-band and observing mode 
    # check that the filter if is N1 or N2 (see table)
    print("Radius of maximum density:", radius_mas)
    # for a SNR of 5 we detect surface brightness of 10 mJy/arcsec**2 (agreeing with the Table of...), radius = 180 would be the circle (middle in the grey dots) -> convert in mas. Can use this method/criterion to the others tests (flux ratio, exptime, size var) -> how do the surface brightness and radius evolve -> most important conclusion
    # double check the method even though good results (especially the one without chop nod)
    
    return brightness_lim, radius_mas

def sensitivity_limit_5_10(snr_image, bgsub_image, Jy_to_ADU_factor, exptime, chop_nod, psf_subtraction, object_name, mode, window_lim):
    print('======================= SNR LIMIT = 5 =======================')
    brightness_lim_5, radius_mas5 = sensitivity_limit(snr_image, bgsub_image, 5, Jy_to_ADU_factor, exptime, chop_nod, psf_subtraction, object_name, window_lim, mode)
    print('======================= SNR LIMIT = 10 =======================')
    brightness_lim_10, radius_mas10 = sensitivity_limit(snr_image, bgsub_image, 10,  Jy_to_ADU_factor, exptime, chop_nod, psf_subtraction, object_name, window_lim, mode)
    return [brightness_lim_5, radius_mas5], [brightness_lim_10, radius_mas10]



#===========================================
#           Complete function
#===========================================

def complete_Scopesim_process(total_flux, flux_ratio, frac_flux_polar, frac_flux_disk,
                              FWHM_polar_x, FWHM_polar_y, FWHM_disk_x, FWHM_disk_y, FWHM_unres, 
                              theta_polar, Dist_source, 
                              pixel_size_mas, d_offset, 
                              object_name, mode, exptime, 
                              window_lim, norm = 'ds9',
                              chop_nod=False, chop_nod_offset=3*u.arcsec,
                              psf_subtraction=False, psf_file='nothing', 
                              IMPLANE_plot=False, READOUT_plot=False, 
                              contour_plot=False, creating_plot_and_FITS=True):
    

    if flux_ratio == 0: 
        print("Flux ratio value invalid")
        return 0
    
    flux_unres = total_flux*flux_ratio 
    flux_extended = total_flux-flux_unres
    flux_disk = flux_extended*frac_flux_disk
    flux_polar = flux_extended*frac_flux_polar
    
    
    theta_disk = theta_polar + 90*u.degree
    Gauss = create_model_3compo(FWHM_polar_x, FWHM_polar_y, flux_polar, theta_polar,
                            FWHM_disk_x, FWHM_disk_y, flux_disk, theta_disk,
                            FWHM_unres, flux_unres, 
                            Dist_source, 
                            pixel_size_mas, d_offset, 
                            mode, object_name, creating_plot_and_FITS, window_lim)
    
    if flux_disk.value == 0 or flux_polar.value == 0 or flux_unres.value == 0:
        model_type = "2_Gaussians"
    else:
        model_type = '3_Gaussians'
            
    file_src = '../models/'+mode+'/'+model_type+'_'+object_name+'.fits'
    file_full_src = file_src
    """SNR_results = SNR_calculation(object_name, file_src,  
                                mode, exptime, window_lim, file_full_src,
                                chop_nod, chop_nod_offset,
                                psf_subtraction, psf_file, 
                                IMPLANE_plot, READOUT_plot, norm,
                                contour_plot, creating_plot_and_FITS)"""
    SNR_results = SNR_calculation(object_name=object_name, file_src=file_src, file_full_src=file_full_src, mode=mode, exptime=exptime, 
                                  window_lim=window_lim, chop_nod=chop_nod, psf_subtraction=psf_subtraction, psf_file=psf_file, 
                                  IMPLANE_plot=IMPLANE_plot, READOUT_plot=READOUT_plot, 
                                  contour_plot=contour_plot, norm=norm, creating_plot_and_FITS=creating_plot_and_FITS)
    
    # SNR_results = [src_readout_data, src_implane, bgsub_implane, bgsub_readout, snr_th_implane, snr_th_readout]
    
    return Gauss, SNR_results


