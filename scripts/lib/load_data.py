import numpy              as np
import matplotlib.pyplot  as plt
import lib.misc_functions as misc
import lib.cap_utils      as cap
from   astropy.io         import fits
#===============================================================================
def load_data(struct,idx):

   # Reading relevant info from config file
   rname      = struct['Runname'][idx]
   rootname   = rname.split('-')[0]
   filename   = rootname+'.fits'
   survey     = struct['Survey'][idx]
   targ_snr   = struct['SNR'][idx]
   snr_min    = struct['SNR_min'][idx]
   lmin       = struct['Lmin'][idx]
   lmax       = struct['Lmax'][idx]
   redshift   = struct['Redshift'][idx]
   velscale   = struct['Velscale'][idx]
   mask_flag  = struct['Mask'][idx]
   porder     = struct['Porder'][idx]
   border     = struct['Border'][idx]
   vmax       = struct['Vmax'][idx]
   mask_width = struct['Mask_width'][idx]
   
   # Survey specific reading of the datacube info
   print(" - Reading the datacube and basic info")
   hdu = fits.open("../data/"+filename)
   if (survey == "SAURON"):
       
       #---------------------------
       hdr  = hdu[1].header
       data = hdu[1].data
       #---------------------------
       spec  = data['DATA_SPE'].T
       espec = np.sqrt(data['STAT_SPE'].T)
       x     = data['XPOS']
       y     = data['YPOS']
       npix  = spec.shape[0]
       nspax = spec.shape[1]
       wave  = hdr['CRVALS'] + hdr['CDELTS']*np.arange(npix)
       psize = 1.0
       
   elif (survey == "MUSE-WFM"):

       if len(hdu) < 3:
          print("ERROR: The MUSE datacube needs 3 extensions: [0] Primary, [1] Data, [2] Variance")
          exit()

       #---------------------------
       hdr   = hdu[1].header
       spec  = hdu[1].data
       espec = np.sqrt(hdu[2].data)
       #---------------------------
       xaxis = np.arange(spec.shape[2])*hdr['CD2_2']*3600.0
       yaxis = np.arange(spec.shape[1])*hdr['CD2_2']*3600.0
       x, y  = np.meshgrid(xaxis,yaxis)
       x, y  = x.ravel(), y.ravel()
       npix  = spec.shape[0]
       nspax = spec.shape[1]*spec.shape[2]
       wave  = hdr['CRVAL3'] + hdr['CD3_3']*np.arange(npix)
       psize = 0.2 # arcsec per spaxel
       
       # Reshaping the 3D cube to 2D
       spec  = np.reshape(spec,(npix,nspax))
       espec = np.reshape(espec,(npix,nspax))
       
   elif (survey == "CALIFA-V1200"):
       
       if len(hdu) < 2:
          print("ERROR: The CALIFA datacube needs 2 extensions: [0] Data, [1] Dispersion")
          exit()

       #---------------------------
       hdr   = hdu[0].header
       spec  = hdu[0].data
       espec = hdu[1].data 
       #---------------------------
       xaxis = np.arange(spec.shape[2])*hdr['CD2_2']*3600.0
       yaxis = np.arange(spec.shape[1])*hdr['CD2_2']*3600.0
       x, y  = np.meshgrid(xaxis,yaxis)
       x, y  = x.ravel(), y.ravel()
       psize = np.abs(x[1]-x[0])
       npix  = spec.shape[0]
       nspax = spec.shape[1]*spec.shape[2]
       wave  = hdr['CRVAL3'] + hdr['CDELT3']*np.arange(npix)

       # Reshaping the 3D cube to 2D
       spec  = np.reshape(spec,(npix,nspax))
       espec = np.reshape(espec,(npix,nspax))

   else:
       
       print("ERROR: Survey not recognised. Please choose between SAURON/MUSE/CALIFA-V1200")
       exit()

   # Correcting the data for redshift
   print(" - Correcting data for redshift")
   wave /= (1.0 + redshift)
   
   # Checking the desired wavelength range is within data wavelength limits
   if (wave[0] > lmin):
       lmin = wave[0]
   if (wave[-1] < lmax):
       lmax = wave[-1]

   # Cutting the data to the desired wavelength range
   print(" - Cutting data to desired wavelength range")
   idx   = (wave >= lmin) & (wave <= lmax)
   wave  = wave[idx]
   spec  = spec[idx,:]
   espec = espec[idx,:]
   npix  = np.sum(idx)

   # Multiplying the data and sigma for some fudge factor so that data > 1
   good = np.isfinite(spec) & (spec > 0.0)
   factor = np.ceil(np.abs(np.amin(np.log10(spec[good]))))
   spec  *= factor
   espec *= factor
      
   # Computing the SNR in each spaxel
   print(" - Computing the SNR of each spaxel")
   signal = np.nanmedian(spec,axis=0)
   noise  = np.abs(np.nanmedian(espec,axis=0)) 
          
   # Selecting those spaxels above SNR_min
   print(" - Selecting spaxels aboove SNR_min")
   idx    = (np.abs((signal/noise)-snr_min) <= 1.0)
   isof   = np.mean(signal[idx])
   idx    = (signal >= isof)
   spec   = spec[:,idx]
   espec  = espec[:,idx]   
   signal = signal[idx]
   noise  = noise[idx]
   x, y   = x[idx], y[idx]
   nspec  = np.sum(idx)
            
   # Determining Voronoi binning to the data
   print(" - Computing the Voronoi binning")
   binNum, xNode, yNode, xBar, yBar, bin_snr, nPixels, scale = cap.voronoi_2d_binning(x, y, \
           signal, noise, targ_snr, plot=False, quiet=True, pixelsize=psize)
      
   print(" - "+str(len(xNode))+" Voronoi bins required")   
      
   # Applying the Voronoi binning to the data
   print(" - Applying the Voronoi binning")
   ubins     = np.unique(binNum)
   nbins     = len(ubins)
   bin_spec  = np.zeros([npix,nbins])
   bin_espec = np.zeros([npix,nbins])
   bin_flux  = np.zeros(nbins)

   for i in range(nbins):
       k = np.where( binNum == ubins[i] )[0]
       valbin = len(k)
       if valbin == 1:
          av_spec     = spec[:,k]
          av_err_spec = espec[:,k]
       else:
          #av_spec     = np.nanmean(spec[:,k],axis=1)
          av_spec     = np.nansum(spec[:,k],axis=1)
          av_err_spec = np.sqrt(np.sum(espec[:,k]**2,axis=1))

       bin_flux[i]    = np.mean(av_spec,axis=0)
       bin_spec[:,i]  = np.ravel(av_spec)
       bin_espec[:,i] = np.ravel(av_err_spec)
       
       misc.printProgress(i+1, nbins, barLength = 50)
       
   # Log-rebinning the data to the input Velscale
   print(" - Log-rebinning and normalizing the spectra")
   lamRange = np.array([np.amin(wave),np.amax(wave)])
   dummy, lwave, _ = cap.log_rebin(lamRange, bin_spec[:,0], velscale=velscale)
   npix_log = len(dummy)
   lspec, lespec = np.zeros([npix_log,nbins]), np.zeros([npix_log,nbins])
   
   for i in range(nbins):
       
      #Log-rebinning the spectra 
      lspec[:,i],  dummy , dummy = cap.log_rebin(lamRange, bin_spec[:,i],  velscale=velscale)
      lespec[:,i], dummy , dummy = cap.log_rebin(lamRange, bin_espec[:,i], velscale=velscale)

      # Normalizing the observed and error spectra respecting the SNR of each bin
      lespec[:,i] /= np.nanmedian(lspec[:,i])
      lspec[:,i]  /= np.nanmedian(lspec[:,i]) 

      misc.printProgress(i+1, nbins, barLength = 50)

   # Defining the mask
   print(" - Defining the data mask")
   if (mask_flag == 1):
     mask = cap.determine_goodpixels(lwave,[lmin,lmax],0.0, width=mask_width, vmax=vmax)
   elif (mask_flag == 0):
     mask = np.arange(npix_log)
      
   # Storing all the info in a data structure
   print(" - Storing everything in data structure")
   print("")
   data_struct = {'binID':      binNum,
                  'x':          x,
                  'y':          y,
                  'flux':       signal,
                  'xbin':       xNode,
                  'ybin':       yNode,
                  'bin_flux':   bin_flux,
                  'spec_obs':   lspec,
                  'sigma_obs':  lespec,
                  'wave_obs':   lwave,
                  'wave':       wave,
                  'velscale':   velscale,
                  'mask':       np.ravel(mask),
                  'mask_width': mask_width,
                  'nmask':      len(mask), 
                  'bin_snr':    bin_snr,
                  'npix':       npix,
                  'npix_obs':   npix_log,
                  'nspec':      nspec,
                  'porder':     porder,
                  'border':     border,
                  'nbins':      nbins,
                  'snr':        targ_snr,
                  'lmin':       lmin,
                  'lmax':       lmax
                 }

   return data_struct

       