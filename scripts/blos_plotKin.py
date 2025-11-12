# -*- coding: utf-8 -*-
"""
    blos_plotKin.py
    Adriano Poci
    University of Oxford
    2025

    <adriano.poci@physics.ox.ac.uk>

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    This module plots the outputs of a BAYES-LOSVD run.

    Author
    ------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.00:  28 March 2025
"""

import os
import sys
import optparse
import warnings
from tqdm import tqdm
import pathlib as plp
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scripts.lib.misc_functions as misc
import seaborn as sns
from astropy.io import ascii

from pxf import putils as pu
from dynamics.IFU.Functions import Geometric

from plotbin.display_pixels import display_pixels as dpp

curdir = plp.Path(plp.Path(__file__).parent).parent
icefire = sns.color_palette("icefire", as_cmap=True)
rocket = sns.color_palette("rocket", as_cmap=True)
GEO = Geometric()

divcmap = 'GECKOSdr'
moncmap = rocket

#==============================================================================
def read_fit(filename, idx, losvd_file=None):

   # Checking bin exists in dataset
   stridx = str(idx)
   hf = h5py.File(filename,"r")
   dummy = hf.get(f"out/{stridx}/bestfit")
   if isinstance(dummy, type(None)):
      misc.printFAILED("ERROR: Bin "+stridx+" does not exist in file")
      sys.exit()

   # Reading input LOSVD if requested
   if not (losvd_file == None): 
      tab = ascii.read(losvd_file)
      input_xvel = tab['col1']
      input_losvd = tab['col2'] / np.sum(tab['col2'])
    
   # Reading the results
   # --- Input data ----------
   xbin     = np.array(hf['in/xbin'])
   ybin     = np.array(hf['in/ybin'])
   wave_obs = np.exp(np.array(hf['in/wave_obs']))
   spec_obs = np.array(hf['in/spec_obs'][:,idx])
   xvel     = np.array(hf['in/xvel'])
   mask     = np.array(hf['in/mask'])
   ndim     = np.array(hf['in/ndim'])
   # --- Output results ---------
   bestfit  = np.array(hf[f"out/{stridx}/bestfit"])
   losvd    = np.array(hf[f"out/{stridx}/losvd"])
   poly     = np.array(hf[f"out/{stridx}/poly"])+1.0
   nbins    = len(xbin)
  
   # Normalizing LOSVDs --------------------------------------------------------
   norm_factor = np.sum(losvd[2,:])
   for i in range(5):
       losvd[i,:] /= norm_factor

   # Bin map -----------
   if ndim > 1:
      ax0 = plt.subplot2grid((2,4),(0,0), colspan=1)
      ax0.set_title("BinID map")
      ax0.plot(xbin,ybin,'k+', zorder=0)
      ax0.plot(xbin[idx],ybin[idx],'r.', markersize=15.0)
      ax0.set_aspect('equal')
      for i in range(nbins):
          ax0.text(xbin[i],ybin[i],i, fontsize=5, horizontalalignment='right', verticalalignment='center',zorder=1)
            
   # LOSVD -----------
   ax1 = plt.subplot2grid((2,4),(0,2), colspan=2)
   ax1.fill_between(xvel,losvd[0,:],losvd[4,:], color='blue', alpha=0.15, step='mid')
   ax1.fill_between(xvel,losvd[1,:],losvd[3,:], color='blue', alpha=0.50, step='mid')
   ax1.plot(xvel,losvd[2,:],'k.-', ds='steps-mid')
   if not (losvd_file == None):
      ax1.plot(input_xvel, input_losvd,'r.-', ds='steps-mid') 
   ax1.axhline(y=0.0,color='k', linestyle='--')
   ax1.axvline(x=0.0, color='k', linestyle=":")
   ax1.set_xlabel("Velocity (km s$^{-1}$)")

   return losvd

# ------------------------------------------------------------------------------

def plotKin(run, PA=0.0):

   filename = curdir/'results'/run/f"{run}_results.hdf5"

   # Reading the results
   hf = h5py.File(filename, 'r')
   xbin = np.array(hf['in/xbin'])
   ybin = np.array(hf['in/ybin'])
   nbins = len(xbin)
   xvel = np.array(hf['in/xvel'])

   xbin, ybin = GEO.rotate2D(xbin*200.0, ybin*200.0, PA)
   xLen, yLen = np.ptp(xbin), np.ptp(ybin) # unmasked pixels

   # Plotting the bins
   fig = plt.figure(figsize=plt.figaspect(yLen/xLen)*2.0)
   ax = fig.gca()
   ax.set_xlim(xbin.min(), xbin.max())
   ax.set_ylim(ybin.min(), ybin.max())
   ax.set_xticklabels(np.array(ax.get_xticks(), dtype=float)/200.0)
   ax.set_yticklabels(np.array(ax.get_yticks(), dtype=float)/200.0)

   LOSVD = np.ma.ones((xbin.size, 5, xvel.size))*np.nan

   lfn = curdir/'results'/run/f"{run}_LOSVD.xz"

   if not lfn.is_file():
      for ji in tqdm(np.arange(xbin.size)):
         # ins = inset_axes(ax, width='100%' ,height='100%',
         #    bbox_to_anchor=((xbin[ji], ybin[ji], 40, 40)),
         #    bbox_transform=ax.transData, axes_class=plt.Axes)
         try:
            # bestfit = np.array(hf[f"out/{ji}/bestfit"])
            losvd = np.array(hf[f"out/{ji}/losvd"])
            # poly = np.array(hf[f"out/{ji}/poly"])+1.0
         #    ins.fill_between(xvel,losvd[0,:],losvd[4,:], color='blue', alpha=0.15, step='mid')
         #    ins.fill_between(xvel,losvd[1,:],losvd[3,:], color='blue', alpha=0.50, step='mid')
         #    ins.plot(xvel, losvd[2,:], '-', lw=0.2)
         #    ins.axhline(y=0.0,color='k', linestyle='--', lw=0.05)
         #    ins.axvline(x=0.0, color='k', linestyle=":", lw=0.05)
         #    ins.set_xticks([])
         #    ins.set_yticks([])
         #    ins.axis('off')

            LOSVD[ji, :, :] = losvd
         except:
            # ins.axis('off')
            continue
      pu.Write.lzma(lfn, LOSVD)
      # fig.savefig(curdir/'results'/run/f"{run}_LOSVD.png", dpi=600)
   else:
      LOSVD = pu.Load.lzma(lfn)
   

   LOSVD = np.ma.masked_invalid(LOSVD)
   # square gridspec with xvel.size panels
   # find square value
   nvel = xvel.size
   ncols = int(np.ceil(np.sqrt(nvel)))
   nrows = int(np.ceil(nvel/ncols))
   # create gridspec
   fig = plt.figure(figsize=plt.figaspect(yLen/xLen)*2.0)
   gs = plt.GridSpec(nrows, ncols, figure=fig)
   # loop over the panels
   vmax = np.mean((np.ma.max(LOSVD[:, 2, :]), 1.0))
   binID = np.array(hf['in/binID'])
   bmask = binID>=0
   binID = binID[bmask]
   for iv in range(nvel):
      ax = fig.add_subplot(gs[iv//ncols, iv%ncols])
      dpp(hf['in/x'][bmask], hf['in/y'][bmask], LOSVD[:, 2, iv][binID], 
         angle=PA, pixelsize=0.2, vmin=0.0, vmax=vmax, cmap=moncmap)
      ax.text(0.5, 1e-3, f"v={xvel[iv]:.1f} km/s",
         horizontalalignment='center', verticalalignment='bottom',
         transform=ax.transAxes, fontsize=8)
      if not ax.get_subplotspec().is_last_row():
            ax.set_xticklabels([])
      if not ax.get_subplotspec().is_first_col():
            ax.set_yticklabels([])
   
   fig.savefig(curdir/'results'/run/f"{run}_channel.png", dpi=600)
   plt.close('all')

#==============================================================================
if (__name__ == '__main__'):

    warnings.filterwarnings("ignore")

    print("===========================================")
    print("               BAYES-LOSVD                 ")
    print("             (inspect_fits)                ")
    print("===========================================")

    # Capturing the command line arguments
    parser = optparse.OptionParser(usage="%prog -f file")
    parser.add_option("-r", "--run",   dest="runname",  type="string", default=None,   help="Runname with the results")
    parser.add_option("-b", "--binID", dest="binID",    type="int",    default=0,      help="ID of the bin to plot")
    parser.add_option("-l", "--losvd", dest="losvd",    type="str",    default=None,   help="(Optional) Filename of the input LOSVD")
    parser.add_option("-s", "--save",  dest="save",     type="int",    default=0,      help="(Optional) Save figure")
    parser.add_option("-d", "--dir",   dest="dir",      type="string", default='../results/', help="(Optional) The directory with results")


    (options, args) = parser.parse_args()
    runname    = options.runname
    binID      = options.binID
    losvd_file = options.losvd
    save       = options.save
    dir        = options.dir
    filename   = dir+runname+"/"+runname+"_results.hdf5"

    if not os.path.exists(filename):
       misc.printFAILED(filename+" does not exist.")
       sys.exit()

    run_inspect_fits(filename,binID,losvd_file,save=save)

    misc.printDONE(runname+" - Bin: "+str(binID))
