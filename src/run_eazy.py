
import astropy.units as u
from typing import Tuple, Union, List, NoReturn, Callable, Dict, Any
from numpy.typing import NDArray
from shutil import copy
import numpy as np
import os
from astropy.table import Table
import itertools
import eazy
from eazy import hdf5

from config import basedir


def extract_filt_profile(
    filter_name: str,
    in_units: u.Quantity = u.um,
    out_units: u.Quantity = u.um,
    filtdir: str = f"{basedir}/inputs/filt",
) -> Tuple[NDArray[float], NDArray[float]]:
    # Given a filter name, extract the filter profile from the EAZY filter file
    filter_path = f"{filtdir}/{filter_name}.txt"
    # read wavelength and transmission from txt file
    wavs, trans = np.loadtxt(filter_path, unpack=True)
    wavs = (wavs * in_units).to(out_units).value
    return wavs, trans


def calc_wav_mean(
    wavs: u.Quantity,
    trans: NDArray[float],
) -> float:
    # Calculate the mean wavelength of a filter
    numerator = np.trapz(wavs * trans, x = wavs)
    denominator = np.trapz(trans, x = wavs)
    return numerator / denominator


def make_filt_file(
    filter_names: Union[List[str], NDArray[str]],
    out_path: str,
    filtdir: str = f"{basedir}/inputs/filt",
    ubvj_filtname: str = "EAZY_UBVJ.RES",
) -> NoReturn:
        '''
        Write a filter file for EAZY

        '''
        # Need to write a filterset file for EAZY
        # Two files one with list of filters and one with the transmission curves
        # format of list is

        # 1 len(transmission_curve) name1 lambda_c = pivot_wav
        # 2 len(transmission_curve) name2 lambda_c = pivot_wav

        # format of transmission curve is
        # len(transmission_curve) name1 lambda_c = pivot_wav
        # 1 0.1 0.0
        # 2 0.2 0.1
        # 3 0.3 0.2
        #...
        # len(transmission_curve) name1 lambda_c = pivot_wav
        # 1 0.1 0.0
        # 2 0.2 0.1

        # copy default filter file and append to it
        ubvj_path = f"{filtdir}/{ubvj_filtname}"
        copy(ubvj_path, out_path)
        copy(f'{ubvj_path}.INFO', f'{out_path}.INFO')
        
        # count lines in .INFO
        with open(f'{out_path}.INFO', 'r') as f:
            current_lines = f.readlines()
            nexisting = len(current_lines)
            last_line = current_lines[-1]

        with open(out_path, 'a') as f:
            with open (f'{out_path}.INFO', 'a') as f_info:
                # work out whether we need to move to the next line - i.e is the current line got anything in it
                
                if not last_line.endswith('\n'):
                    f_info.write('\n')

                f.write('\n')

                # count lines in file
                for i, filt_name in enumerate(filter_names):
                    code = i + nexisting + 1
                    wavs, trans = extract_filt_profile(filt_name, in_units = u.um, out_units = u.um)
                    wav_mean = calc_wav_mean(wavs, trans)
                    f_info.write(f'{code}  {len(trans)} {filt_name.replace("/", ".")} lambda_c= {wav_mean}\n')
                    f.write(f' {len(trans)} {filt_name.replace("/", ".")} lambda_c= {wav_mean}\n')

                    for pos, (wav_, trans_) in enumerate(zip(wavs, trans)):
                        f.write(f'{pos + 1} {wav_} {trans_}\n')
                        
        return np.arange(nexisting + 1, nexisting + 1 + len(filter_names))


def galfind_cat_load_phot_func(
    tab: Table,
    filter_names: Union[List[str], NDArray[str]],
    out_units: Union[u.Magnitude, u.Quantity] = u.uJy,
) -> Tuple[NDArray[float], NDArray[float]]:
    # Load photometry from a GalFind catalogue Table
    n_gals = len(tab)
    n_filters = len(filter_names)
    phot = np.zeros((n_gals, n_filters))
    phot_errs = np.zeros((n_gals, n_filters))

    for i, filt_name in enumerate(filter_names):
        band_name = filt_name.split("/")[1]
        phot[:, i] = tab[f"FLUX_APER_{band_name}_aper_corr_Jy"]
        phot_errs[:, i] = tab[f"FLUXERR_APER_{band_name}_loc_depth_10pc_Jy"]

    # you will need to insert -99s for garbage or non-existent data here

    phot = (phot * u.Jy).to(out_units).value
    phot_errs = (phot_errs * u.Jy).to(out_units).value
    return phot, phot_errs


def make_phot_file(
    cat_path: str,
    filter_names: Union[List[str], NDArray[str]],
    filter_codes: NDArray[int],
    out_path: str,
    load_phot_func: Callable = galfind_cat_load_phot_func,
    hdu: str = "OBJECTS",
    out_units: str = u.uJy,
) -> NoReturn:
    # load photometry from catalogue
    tab = Table.read(cat_path, hdu = "OBJECTS")
    IDs = tab["ASCENDING_ID"].data
    # set all redshifts to unknown
    redshifts = np.full(len(tab), -99.0)
    phot, phot_errs = load_phot_func(tab, filter_names, out_units = out_units)
    # Make input file
    data = np.array(
        [
            np.concatenate(
                (
                    [IDs[i]],
                    list(itertools.chain(*zip(phot[i], phot_errs[i]))),
                    [redshifts[i]],
                ),
                axis=None,
            )
            for i in range(len(IDs))
        ]
    )
    names = (
        ["ID"]
        + list(
            itertools.chain(
                *zip(
                    [f"F{filter_code}" for filter_code in filter_codes],
                    [f"E{filter_code}" for filter_code in filter_codes],
                )
            )
        )
        + ["z_spec"]
    )
    types = (
        [int]
        + list(np.full(len(filter_names) * 2, float))
        + [float]
    )
    out_tab = Table(data, dtype=types, names=names)
    out_tab.write(
        out_path,
        format="ascii.commented_header",
        delimiter=" ",
        overwrite=True,
    )


def perform_fit(
    in_cat_path: str,
    out_cat_path: str,
    filt_path: str,
    phot_path: str,
    template_set: str = "fsps_larson",
    default_param_path: str = f"{basedir}/inputs/zphot.param.default",
    default_wav_path : str = f"{basedir}/inputs/templates/lambda.def",
    default_temp_err_file: str = f"{basedir}/inputs/templates/TEMPLATE_ERROR.eazy_v1.0",
    default_translate_file: str = f"{basedir}/inputs/templates/zphot_jwst.translate",
    save_ubvj: bool = True,
    n_proc: int = 1,
) -> NoReturn:
    fit_params = {}
    fit_params["FILTERS_RES"] = filt_path
    fit_params["WAVELENGTH_FILE"] = default_wav_path # Wavelength grid definition file
    fit_params["TEMP_ERR_FILE"] = default_temp_err_file # Template error definition file
    fit_params["CATALOG_FILE"] = in_cat_path
    fit_params["MAIN_OUTPUT_FILE"] = out_cat_path
    fit_params["OUTPUT_DIRECTORY"] = "/".join(out_cat_path.split("/")[:-1])
    fit_params["TEMPLATES_FILE"] = f"{basedir}/inputs/templates/{template_set}/{template_set}.param"

    fit = eazy.photoz.PhotoZ(
        param_file=default_param_path,
        zeropoint_file=None,
        params=fit_params,
        load_prior=False,
        load_products=False,
        translate_file=default_translate_file,
        n_proc=n_proc,
    )
    fit.fit_catalog(n_proc=n_proc, get_best_fit=True)
    # Save backup of fit in hdf5 file
    h5_path = out_cat_path.replace(".fits", ".h5")
    hdf5.write_hdf5(
        fit,
        h5file=h5_path,
        include_fit_coeffs=False,
        include_templates=True,
        verbose=False,
    )
    data = {
        "IDENT": fit.OBJID,
        "zbest": fit.zbest,
        "zbest_16": fit.pz_percentiles([16]),
        "zbest_50": fit.pz_percentiles([50]),
        "zbest_84": fit.pz_percentiles([84]),
        "chi2_best": fit.chi2_best,
    }
    tab = Table(data)

    # Get rest frame colors
    if save_ubvj:
        # This is all duplicated from base code.
        rf_tempfilt, lc_rest, ubvj = fit.rest_frame_fluxes(
            f_numbers=[1, 2, 3, 4], simple=False, n_proc=n_proc
        )
        for i, ubvj_filt in enumerate(["U", "B", "V", "J"]):
            tab[f"{ubvj_filt}_rf_flux"] = ubvj[:, i, 2]
            # symmetric errors
            tab[f"{ubvj_filt}_rf_flux_err"] = (
                ubvj[:, i, 3] - ubvj[:, i, 1]
            ) / 2.0

    # add the template name to the column labels except for IDENT
    for col_name in tab.colnames:
        if col_name != "ASCENDING_ID":
            tab.rename_column(
                col_name,
                f"{col_name}_{template_set}",
            )
    # Write fits file
    tab.write(out_cat_path, overwrite=True)


def main(
    filter_names: Union[List[str], NDArray[str]],
    in_cat_path: str,
    out_cat_path: str,
    filt_path: str,
    phot_path: str,
    template_set: str = "fsps_larson",
) -> NoReturn:
    os.makedirs(os.path.dirname(filt_path), exist_ok=True)
    filter_codes = make_filt_file(filter_names, filt_path)
    os.makedirs(os.path.dirname(phot_path), exist_ok=True)
    if not os.path.exists(phot_path):
        make_phot_file(in_cat_path, filter_names, filter_codes, phot_path)
    os.makedirs(os.path.dirname(out_cat_path), exist_ok=True)
    if not os.path.exists(out_cat_path):
        perform_fit(phot_path, out_cat_path, out_filt_path, out_phot_path, template_set)


if __name__ == "__main__":
    # change these to the bands you want to use for SED fitting
    filter_names = ["NIRCam/F150W", "NIRCam/F200W", "NIRCam/F277W"]
    in_cat_path = f"{basedir}/inputs/cats/EPOCHS_JADES-GS.fits"
    out_cat_path = in_cat_path.replace("inputs", "outputs")
    out_filt_path = out_cat_path.replace("cats", "filt").replace(".fits", ".RES")
    out_phot_path = out_cat_path.replace("cats", "phot").replace(".fits", ".in")
    template_set = "fsps_larson"
    main(filter_names, in_cat_path, out_cat_path, out_filt_path, out_phot_path, template_set)