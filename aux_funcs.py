import os
import astropy.units as u
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.coordinates import Angle, SkyCoord
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from astropy.coordinates.name_resolve import NameResolveError


# ---- RA tick formatter (degrees → HMS) ----
def ra_formatter(x, pos):
    return Angle(x, unit=u.deg).to_string(
        unit=u.hour,
        sep=':',
        precision=1,
        pad=True
    )


# ---- DEC tick formatter (degrees → DMS) ----
def dec_formatter(y, pos):
    return Angle(y, unit=u.deg).to_string(
        unit=u.deg,
        sep=':',
        precision=1,
        alwayssign=True,
        pad=True
    )


def query_images_from_legacy_survey(
    ra_pointing,
    dec_pointing,
    cluster_name,
    output_dir,
    image_size_pix=1400,
    layer='ls-dr10',
    pixscale=0.05,
    bands='g'
):

    """
    Download a FITS cutout image from the Legacy Survey viewer.

    Parameters
    ----------
    ra_pointing : float
        Right Ascension of the field center in degrees (ICRS).

    dec_pointing : float
        Declination of the field center in degrees (ICRS).

    cluster_name : str
        Name used to construct the output filename.

    output_dir : str
        Directory where the FITS image will be saved.

    image_size_pix : int, optional
        Image size in pixels (default is 1400).

    layer : str, optional
        Legacy Survey data release layer (default is 'ls-dr10').

    pixscale : float, optional
        Pixel scale in arcseconds per pixel (default is 0.05).

    bands : str, optional
        Photometric band(s) to retrieve (default is 'g').
        Possible inputs are 'ugriz'.

    Returns
    -------
    None
    """

    url = 'https://www.legacysurvey.org/viewer/fits-cutout?ra={}&dec={}&size={}&layer={}&pixscale={}&bands={}'.format(
        ra_pointing,
        dec_pointing,
        image_size_pix,
        layer,
        pixscale,
        bands
    )

    cmd = 'wget -O {}/{}_{}_{}.fits "{}"'.format(
        output_dir,
        cluster_name,
        ra_pointing,
        dec_pointing,
        url
    )

    os.system(command=cmd)
    print(cmd)
    print(
        f'Image saved as: {cluster_name}_{ra_pointing}_{dec_pointing}.fits'
    )

    return


def query_gaia_dr3_region(
    ra_pointing,
    dec_pointing,
    output_dir,
    targname=None,
    cluster_name='region',
    radius_arcmin=5.0,
):
    """
    Query Gaia DR3 sources around a given sky position.

    Parameters
    ----------
    ra_pointing : float, optional
        Right Ascension in degrees (ICRS).

    dec_pointing : float, optional
        Declination in degrees (ICRS).

    output_dir : str, optional
        Directory where Gaia catalog CSV will be saved.

    targname : str, optional
        Target name to resolve if RA/DEC are not provided.
        Uses `SkyCoord.from_name()`.

    cluster_name : str, optional
        Name used for output filename. Default is "region".

    radius_arcmin : float, optional
        Search radius in arcminutes. Default is 5.0.

    Returns
    -------
    pandas.DataFrame
        Gaia DR3 query results as a pandas DataFrame.

    Raises
    ------
    ValueError
        If neither coordinates nor targname are provided.
    """

    print("--- QUERYING GAIA REGION ---")

    # --------------------------------------------------
    # Resolve coordinates if necessary
    # --------------------------------------------------
    if (ra_pointing is None) or (dec_pointing is None):

        if targname is None:
            raise ValueError(
                "You must provide either (ra_pointing, dec_pointing) "
                "or a targname to resolve coordinates."
            )

        print(f"Resolving coordinates for target: {targname}")
        coord = SkyCoord.from_name(targname)
        ra_pointing = coord.ra.deg
        dec_pointing = coord.dec.deg

    print(f"RA  (deg): {ra_pointing}")
    print(f"DEC (deg): {dec_pointing}")
    print(f"Radius (arcmin): {radius_arcmin}")

    # --------------------------------------------------
    # Build ADQL query
    # --------------------------------------------------
    query = f"""
    SELECT
        source_id,
        ra,
        dec,
        pmra,
        pmdec,
        phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE
        CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra_pointing}, {dec_pointing}, {radius_arcmin}/60.0)
        ) = 1
    """

    # --------------------------------------------------
    # Execute query
    # --------------------------------------------------
    job = Gaia.launch_job_async(query)
    results = job.get_results()

    print(f"Number of sources found: {len(results)}")
    print(f"Columns: {results.colnames}")

    # --------------------------------------------------
    # Convert and save
    # --------------------------------------------------
    df_gaia_catalog = results.to_pandas()

    os.makedirs(output_dir, exist_ok=True)
    csv_out_path = os.path.join(output_dir, f"{cluster_name}_gaia.csv")

    df_gaia_catalog.to_csv(csv_out_path, index=False)
    print(f"Catalog saved to: {csv_out_path}\n")

    return df_gaia_catalog


def query_hsc_cone(
    ra_pointing,
    dec_pointing,
    radius_arcmin,
    version="v3"
):

    """
    Query the Hubble Source Catalog (HSC) around a sky position.

    Parameters
    ----------
    ra_pointing : float
        Right Ascension in degrees (ICRS).

    dec_pointing : float
        Declination in degrees (ICRS).

    radius_arcmin : float
        Search radius in arcminutes.

    version : str, optional
        HSC version to query (default is "v3").

    Returns
    -------
    pandas.DataFrame
        HSC sources within the search radius. Returns an empty
        DataFrame if no sources are found.

    Notes
    -----
    This function queries the HSC via MAST using
    astroquery.mast.Catalogs.query_region.
    """

    # Explicit ICRS coordinates
    coord = SkyCoord(
        ra=ra_pointing * u.deg,
        dec=dec_pointing * u.deg,
        frame="icrs"
    )

    radius = radius_arcmin * u.arcmin

    # Query HSC via MAST
    hsc_table = Catalogs.query_region(
        coord,
        radius=radius,
        catalog="HSC",
        version=version
    )

    if len(hsc_table) == 0:
        return pd.DataFrame()

    return hsc_table.to_pandas().reset_index(drop=True)


def get_coords_from_targname(targname):
    """
    Resolve sky coordinates (RA, DEC in degrees) from a target name.

    Parameters
    ----------
    targname : str
        Astronomical target name resolvable via online name services
        (e.g., SIMBAD, NED).

    Returns
    -------
    tuple of float
        (ra_deg, dec_deg) in degrees.

    Raises
    ------
    ValueError
        If the target name cannot be resolved or is invalid.
    """

    if not targname or not isinstance(targname, str):
        raise ValueError("targname must be a non-empty string.")

    print(f"Resolving coordinates for target: {targname}")

    try:
        coord = SkyCoord.from_name(targname)
    except NameResolveError:
        raise ValueError(
            f"Could not resolve coordinates for target '{targname}'. "
            "Check the spelling or try a different identifier."
        )
    except Exception as e:
        raise ValueError(
            f"Unexpected error while resolving '{targname}': {e}"
        )

    ra_pointing = coord.ra.deg
    dec_pointing = coord.dec.deg

    if (ra_pointing is None) or (dec_pointing is None):
        raise ValueError(
            f"Resolved target '{targname}' but coordinates were invalid."
        )

    print(f"RA  (deg): {ra_pointing}")
    print(f"DEC (deg): {dec_pointing}")

    return ra_pointing, dec_pointing


def append_catalog_table_to_pipeline_table(
    field_csv_file_path,
    extname,
    ra_pointing_deg,
    ra_pointing_hms,
    dec_pointing_deg,
    dec_pointing_dms,
    catalog_path,
    pipeline_table_fits='MU_GASR_141026A_astrom_cat.fits'
):
    """
    Append a CSV catalog as a BinTableHDU to the existing FITS pipeline file.

    The function reads a CSV file containing candidate sources,
    converts it into a FITS binary table, attaches metadata
    (field coordinates and catalog path), and appends it as a
    new extension to the existing FITS file.

    Parameters
    ----------
    field_csv_file_path: str
        Path to the input CSV file containing the field catalog.
        The catalog must have the following columns:
            - SourceID (identifier for the object)
            - RA (in degrees)
            - DEC (in degrees)
            - filter (of the magnitude, F555W, F606W, Vvega, ...)
            - mag (magnitude)

    extname : str
        EXTNAME header value for the new FITS extension.
        Example for WFM: "NGC_1851_f01"
        Example for NFM: "NGC5986_f028_NFM"

    ra_pointing_deg : float
        Field pointing Right Ascension (degrees), stored in header.

    ra_pointing_hms : str
        Field pointing Right Ascension (HMS), stored in header comment.

    dec_pointing_deg : float
        Field pointing Declination (degrees), stored in header.

    dec_pointing_dms : str
        Field pointing Declination (DMS), stored in header comment.

    catalog_path : str
        Path to the original catalog from which the field was extracted.

    pipeline_table_fits : str, optional
        FITS file to append the new table to
        (default is 'MU_GASR_141026A_astrom_cat.fits').

    Returns
    -------
    astropy.io.fits.HDUList
        The updated FITS HDUList after appending the new extension.

    Notes
    -----
    - Byte-string artifacts (e.g., b'...') are removed from
      'SourceID' and 'filter' columns before writing, which are
      present in the catalog outputs from the previous functions.
    - The SourceID column is truncated to 8 characters.
    - A CHECKSUM keyword is added to the new extension.
    """

    # --------------------------------------------------
    # Helper: clean byte strings
    # --------------------------------------------------
    def clean_byte_string(series):
        return (
            series.astype(str)
            .str.replace(r"^b'", "", regex=True)
            .str.replace(r"'$", "", regex=True)
        )

    df = pd.read_csv(field_csv_file_path)

    df["SourceID"] = clean_byte_string(df["SourceID"])
    df["filter"] = clean_byte_string(df["filter"])

    sourceid = df["SourceID"].str.slice(0, 8).to_numpy(dtype="S8")

    ra = df["RA"].astype(np.float64).to_numpy()
    dec = df["DEC"].astype(np.float64).to_numpy()
    mag = df["mag"].astype(np.float64).to_numpy()

    max_filter_len = df["filter"].str.len().max()
    filter_format = f"{max_filter_len}A"
    filt = df["filter"].to_numpy(dtype=f"S{max_filter_len}")

    cols = fits.ColDefs([

        fits.Column(
            name="SourceID",
            format="8A",
            array=sourceid
        ),

        fits.Column(
            name="RA",
            format="1D",
            unit="deg",
            array=ra
        ),

        fits.Column(
            name="DEC",
            format="1D",
            unit="deg",
            array=dec
        ),

        fits.Column(
            name="filter",
            format=filter_format,
            array=filt
        ),

        fits.Column(
            name="mag",
            format="1D",
            unit="mag",
            array=mag
        )
    ])

    hdu = fits.BinTableHDU.from_columns(cols)
    hdr = hdu.header

    hdr["EXTNAME"] = (extname, "field name")
    hdr["RA"] = (float(ra_pointing_deg), ra_pointing_hms)
    hdr["DEC"] = (float(dec_pointing_deg), dec_pointing_dms)
    hdr["CATALOG"] = (catalog_path, "Input catalog")

    hdu.add_checksum()

    with fits.open(pipeline_table_fits, mode="append") as hdul:
        hdul.append(hdu)

    return hdul
