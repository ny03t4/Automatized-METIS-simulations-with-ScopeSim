from matplotlib.patches import Circle

snr_readout = Circinus_1_10[1][5]
pxl = np.where((snr_readout < 6) & (snr_readout >4))
#print(np.size(pxl))
#print(np.size(snr_readout))
# find out how is the pixel size in this image compared to our resolution
bgsub_readout = Circinus_1_10[1][3]/Jy_to_ADU_chopnod

mask = np.zeros_like(snr_readout, dtype=bool)

mask[pxl] = True 
#plt.imshow(mask)
center = np.shape(bgsub_readout)[0]/2

bgsub_readout_masked = np.where(mask, bgsub_readout, np.nan)

imsrcraw=plt.imshow(bgsub_readout_masked)
plt.colorbar(imsrcraw)
ax= plt.gca()
ax.add_patch(Circle([center, center], 100, fill=False, edgecolor='black'))
ax.add_patch(Circle([center, center], 200, fill=False, edgecolor='red'))
#print(bgsub_readout_masked)

sum_masked = np.nansum(bgsub_readout_masked)
print(sum)
area_1pxl = N_band_pixel_scale**2
area_mask = np.size(pxl)*area_1pxl

print(sum_masked*u.Jy/area_mask)
a = sum_masked*u.Jy/area_mask
a = a.to(u.mJy/u.arcsec**2)
print(a)
# find out what is the sensitivity of metis in N-band and observing mode 
# check that the filter if is N1 or N2 (see table)

print(200*N_band_pixel_scale)
# for a SNR of 5 we detect surface brightness of 10 mJy/arcsec**2 (agreeing with the Table of...), radius = 180 would be the circle (middle in the grey dots) -> convert in mas. Can be used this method/criterion to the others tests (flux ratio, exptime, size var) -> how do the surface brightness and radius evolve -> most important conclusion
# double check the method even though good results (especially the one without chop nod)