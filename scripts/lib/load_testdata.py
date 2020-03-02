import numpy              as np
import matplotlib.pyplot  as plt
import lib.misc_functions as misc
import lib.cap_utils      as cap
from   astropy.io         import fits
#===============================================================================
def load_testdata(struct,idx):

   # Reading relevant info from config file
   rname      = struct['Runname'][idx]
   rootname   = rname.split('-')[0]
   filename   = rootname+'.fits'
   targ_snr   = struct['SNR'][idx]
   lmin       = struct['Lmin'][idx]
   lmax       = struct['Lmax'][idx]
   porder     = struct['Porder'][idx]
   border     = struct['Border'][idx]
   vmax       = struct['Vmax'][idx]
   mask_width = struct['Mask_width'][idx]
   velscale   = struct['Velscale'][idx]
   
   # Survey specific reading of the datacube info
   print(" - Reading the datacube and basic info")
   hdu = fits.open("../data/"+filename)
   #---------------------------
   hdr  = hdu[0].header
   data = hdu[1].data
   #---------------------------
   lspec  = data['SPEC']
   lespec = data['ESPEC']
   lwave  = data['WAVE']
   npix   = lspec.shape[0]
   nspec  = 1
   nbins  = 1

   # Checking the desired wavelength range is within data wavelength limits
   if (np.exp(lwave[0]) > lmin):
       lmin = np.exp(lwave[0])
   if (np.exp(lwave[-1]) < lmax):
       lmax = np.exp(lwave[-1])

   # Cutting the data to the desired wavelength range
   print(" - Cutting data to desired wavelength range")
   idx   = (np.exp(lwave) >= lmin) & (np.exp(lwave) <= lmax)
   lwave  = lwave[idx]
   lspec  = lspec[idx]
   lespec = lespec[idx]
   npix  = np.sum(idx)
      
   # SNR 
   bin_snr = hdr['SNR']
            
#    # Log-rebinning the data to the input Velscale
#    print(" - Log-rebinning and normalizing the spectra")
#    lamRange = np.array([np.amin(wave),np.amax(wave)])
#    lspec,  lwave,   _    = cap.log_rebin(lamRange, spec,  velscale=velscale)
#    lespec, dummy , dummy = cap.log_rebin(lamRange, espec, velscale=velscale)
   npix_log = len(lspec)

   # Normalizing the observed and error spectra respecting the SNR of each bin
   lespec /= np.nanmedian(lspec)
   lspec  /= np.nanmedian(lspec) 

   # Defining the mask
   print(" - Defining the data mask")
   mask = cap.determine_goodpixels(lwave,[lmin,lmax],0.0, width=mask_width, vmax=vmax)
   mask = np.arange(mask[0],mask[-1])
      
   # Storing all the info in a data structure
   print(" - Storing everything in data structure")
   print("")
   data_struct = {'binID':     np.zeros(1),
                  'x':         np.zeros(1),
                  'y':         np.zeros(1),
                  'flux':      np.ones(1)*np.mean(lspec),
                  'xbin':      np.zeros(1),
                  'ybin':      np.zeros(1),
                  'bin_flux':  np.ones(1)*np.mean(lspec),
                  'spec_obs':  lspec.reshape(len(lspec),1),
                  'sigma_obs': lespec.reshape(len(lespec),1),
                  'wave_obs':  lwave,
                #   'wave':      wave,
                  'velscale':  velscale,
                  'mask':      np.ravel(mask),
                  'nmask':     len(mask),
                  'mask_width': mask_width,
                  'bin_snr':   bin_snr,
                  'npix':      npix,
                  'npix_obs':  npix_log,
                  'nspec':     nspec,
                  'porder':    porder,
                  'border':    border,
                  'nbins':     nbins,
                  'snr':       targ_snr,
                  'lmin':      lmin,
                  'lmax':      lmax
                 }

   return data_struct

       