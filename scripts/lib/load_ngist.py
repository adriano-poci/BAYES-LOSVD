# -*- coding: utf-8 -*-
"""
    load_ngist.py
    Adriano Poci
    University of Oxford
    2025

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    This module reads in the data products of `ngist` in order to fit them with
        BAYES-LOSVD.

    Authors
    -------
    Adriano Poci <adriano.poci@physics.ox.ac.uk>

History
-------
v1.0:   25 March 2025
v1.1:   Use `lmin` and `lmax` from BAYES-LOSVD configuration file instead of
            nGIST to allow for post-Voronoi-binning spectral truncation. 9 April
            2025
"""

import sys, toml, yaml
import pathlib as plp
import numpy as np
from astropy.io import fits
import importlib.util
import lib.misc_functions as misc
import lib.cap_utils as cap

bidx = plp.Path(__file__).parts.index('BAYES-LOSVD')
based = plp.Path(*plp.Path(__file__).parts[:bidx+1])

def load_ngist(struct, config_filename):
    """
    Load NGIST data and configuration, process spectral data, and return a structured dictionary.
    Parameters:
    -----------
    struct : dict
        A dictionary containing the following keys:
        - 'redshift' (float): The redshift value for the data.
        - 'instrument' (str): The name of the instrument used for observation.
        - 'lmin' (float): The minimum wavelength to consider.
        - 'lmax' (float): The maximum wavelength to consider.
        - 'mask_file' (str): The filename of the spectral mask file (or "None" if no mask is used).
        - 'porder' (int): The polynomial order for fitting.
    config_filename : str
        Path to the NGIST configuration file.
    Returns:
    --------
    dict
        A dictionary containing processed data and metadata, with the following keys:
        - 'binID' (numpy.ndarray): Bin IDs for each spaxel.
        - 'x' (numpy.ndarray): X pixel positions.
        - 'y' (numpy.ndarray): Y pixel positions.
        - 'flux' (numpy.ndarray): Flux values for each spaxel.
        - 'xbin' (numpy.ndarray): X positions of bin centers.
        - 'ybin' (numpy.ndarray): Y positions of bin centers.
        - 'bin_flux' (numpy.ndarray): Mean flux values for each bin.
        - 'spec_obs' (numpy.ndarray): Observed spectra for each bin.
        - 'sigma_obs' (numpy.ndarray): Observed spectral uncertainties for each bin.
        - 'wave_obs' (numpy.ndarray): Log-rebinned wavelength array.
        - 'wave' (numpy.ndarray): Observed wavelength array (corrected for redshift).
        - 'velscale' (float): Velocity scale of the spectra.
        - 'mask' (numpy.ndarray): Mask array for valid spectral pixels.
        - 'nmask' (int): Number of valid spectral pixels in the mask.
        - 'bin_snr' (numpy.ndarray): Signal-to-noise ratio for each bin.
        - 'npix' (int): Number of spectral pixels in the original data.
        - 'npix_obs' (int): Number of log-rebinned spectral pixels.
        - 'nspec' (int): Total number of spaxels.
        - 'porder' (int): Polynomial order for fitting.
        - 'nbins' (int): Number of bins.
        - 'snr' (float): Target signal-to-noise ratio for spatial binning.
        - 'lmin' (float): Minimum wavelength considered.
        - 'lmax' (float): Maximum wavelength considered.
        - 'ndim' (int): Dimensionality of the data (always 2).
    Raises:
    -------
    AssertionError
        If the redshift value in `struct` does not match the value in the configuration file.
    ValueError
        If the specified instrument is not found in the instrument configuration file.
    FileNotFoundError
        If the instrument read file or mask file is not found.
    Notes:
    ------
    - The function reads NGIST configuration and data products, processes spectral data, and normalizes it.
    - It also applies a spectral mask if specified.
    - The returned dictionary is structured for use in BAYES-LOSVD analysis.
    """


    # Read ngist configuration
    config_filename = plp.Path(config_filename)
    with open(config_filename, 'r') as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)

    # assert struct['velscale'] == config['PREPARE_SPECTRA']['VELSCALE'],\
    #     'Velocity scale values in BAYES-LOSVD and NGIST configurations do not '\
    #     f"match; {struct['velscale']} (BL) != "\
    #     f"{config['PREPARE_SPECTRA']['VELSCALE']} (NG)"
    assert struct['redshift'] == config['GENERAL']['REDSHIFT'], 'Redshift'\
        ' values in BAYES-LOSVD and NGIST configurations do not match; '\
        f"{struct['redshift']} (BL) != {config['GENERAL']['REDSHIFT']} (NG)"

    instr_config = toml.load(based/'config_files'/'instruments.properties')
    instr_list = list(instr_config.keys())
    # get instrument from config
    instrName = struct['instrument']
    if instrName not in instr_list:
        raise ValueError(f"Instrument '{instrName}' not found in instruments "\
            "configuration file")
    if not (based/'config_files'/'instruments'/instr_config[instrName]['read_file']).is_file():
        raise FileNotFoundError(f"Instrument read file "\
            f"'{instr_config[instrName]['read_file']}' not found in instruments"\
            "directory")
    
    instr = importlib.util.spec_from_file_location("", based/'config_files'/'instruments'/instr_config[instrName]['read_file'])
    module = importlib.util.module_from_spec(instr)
    instr.loader.exec_module(module)

    # table.fits
    with fits.open(plp.Path(config_filename.parent/\
        f"{config["GENERAL"]["RUN_ID"]}_table.fits")) as hdu:
        xpix = np.array(hdu[1].data.X)
        ypix = np.array(hdu[1].data.Y)
        binID = np.array(hdu[1].data.BIN_ID)
        bidx = np.unique(binID, return_index=True)[1]
        xbin = np.array(hdu[1].data.XBIN)[bidx] # bin centres only
        ybin = np.array(hdu[1].data.YBIN)[bidx]
        bin_snr = np.array(hdu[1].data.SNRBIN)[bidx]
        signal = np.array(hdu[1].data.FLUX)

    # Read ngist data products
    with fits.open(plp.Path(config_filename.parent/\
        f"{config["GENERAL"]["RUN_ID"]}_BinSpectra_linear.fits")) as hdu:
        wave = np.array(hdu[2].data.LOGLAM)
        # ``logLam''from _linear is actually linear
    with fits.open(plp.Path(config_filename.parent/\
        f"{config["GENERAL"]["RUN_ID"]}_BinSpectra.fits")) as hdu:
        bin_data = np.array(hdu[1].data.SPEC.T)
        bin_err = np.array(hdu[1].data.ESPEC.T)
        lwave = np.array(hdu[2].data.LOGLAM)
        velscale = hdu[0].header["VELSCALE"]
    idx_lam = np.where(
        np.logical_and(
            lwave > np.log(struct['lmin']),
            lwave < np.log(struct['lmax']),
        )
    )[0]
    # wave = wave[idx_lam]

    # BAYES-LOSVD variables
    lwave = lwave[idx_lam]
    spec = bin_data[idx_lam, :]
    espec = bin_err[idx_lam, :]
    # normalise
    relErr = np.divide(espec, spec)
    spec = np.divide(spec, np.ma.median(spec, axis=0))
    # invidiual spectra medians
    espec = np.abs(spec)*relErr
    wave /= (1.0 + config["GENERAL"]["REDSHIFT"])
    npix = bin_data.shape[0]
    nbins = bin_data.shape[1]
    ubins = np.arange(0, nbins)

    # lamRange = np.array([np.amin(wave), np.amax(wave)])
    # dummy, lwave, _ = cap.log_rebin(lamRange, spec[:,0], velscale=struct['velscale'])
    npix_log = spec[:, 0].size

    # Defining the data mask
    print(" - Defining the data mask")
    if (struct['mask_file'] == "None"):
        mn   = np.int(0.01*npix_log) # Masking edges only
        mask = np.arange(mn, npix_log-mn)
    else:
        if not (based/'config_files'/struct['mask_file']).is_file():
            misc.printFAILED("Cannot find mask file in 'config_files' directory")
            sys.exit()
        mask = misc.spectralMasking(based/'config_files'/struct['mask_file'],
            lwave, struct['redshift'])

    data_struct = dict(
        binID=binID,
        x=xpix,# pixel positions
        y=ypix,
        flux=signal,
        xbin=xbin, # bin positions
        ybin=ybin,
        bin_flux=np.nanmean(spec, axis=0),
        spec_obs=spec,
        sigma_obs=espec,
        wave_obs=lwave, # log-rebinned wavelength
        wave=wave, # observed wavelength
        velscale=velscale,
        mask=np.ravel(mask), # mask
        nmask=len(mask),
        bin_snr=bin_snr, # bin signal-to-noise ratio
        npix=npix,
        npix_obs=npix_log, # number of log-rebinned spectral pixels
        nspec=signal.size, # total number of spaxels
        porder=struct['porder'], # polynomial order
        nbins=nbins,
        snr=config['SPATIAL_BINNING']['TARGET_SNR'],# S/N threshold for binning
        lmin=struct['lmin'], # minimum wavelength
        lmax=struct['lmax'], # maximum wavelength
        ndim=2
    )

    return data_struct