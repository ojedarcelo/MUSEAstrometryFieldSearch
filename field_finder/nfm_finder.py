import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from aux_funcs import ra_formatter, dec_formatter


plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})


def find_reference_stars(
    catalog,
    ra_colname,
    dec_colname,
    mag_colname,
    refstar_mag_lower_lim,
    refstar_mag_upper_lim
):
    """
    Identify potential reference stars within a magnitude range.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Input source catalog containing sky coordinates and magnitudes.

    ra_colname : str
        Name of the Right Ascension column (in degrees).

    dec_colname : str
        Name of the Declination column (in degrees).

    mag_colname : str
        Name of the magnitude column.

    refstar_mag_lower_lim : float
        Lower magnitude limit (brighter bound).

    refstar_mag_upper_lim : float
        Upper magnitude limit (fainter bound).

    Returns
    -------
    pandas.DataFrame
        Subset of the input catalog containing only sources with
        magnitudes between the specified limits.

    Notes
    -----
    Prints the number of identified reference stars and their
    coordinates and magnitudes.

    The magnitude condition applied is:

        refstar_mag_lower_lim <= mag <= refstar_mag_upper_lim
    """

    refstars_catalog = catalog[
        (catalog[mag_colname] <= refstar_mag_upper_lim) & (catalog[mag_colname] >= refstar_mag_lower_lim)
    ]

    print("NFM: Found {} reference star(s)".format(refstars_catalog.shape[0]))
    for index, row in refstars_catalog.iterrows():
        print("    Reference star, RA: {}, DEC: {}, {}: {}".format(
            row[ra_colname], row[dec_colname], mag_colname, row[mag_colname])
        )

    return refstars_catalog


def find_fields_with_reference_star(
    catalog,
    ra_ref_deg,
    dec_ref_deg,
    ra_colname,
    dec_colname,
    mag_colname,
    id_colname,
    filter_name=None,
    half_width_arcsec=7.5/2
):
    """
    Extract a small square field centered on a given reference star.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Full source catalog.

    ra_ref_deg : float
        Right Ascension of the reference star (degrees).

    dec_ref_deg : float
        Declination of the reference star (degrees).

    ra_colname : str
        Name of the RA column (degrees).

    dec_colname : str
        Name of the DEC column (degrees).

    mag_colname : str
        Name of the magnitude column.

    id_colname : str
        Name of the source identifier column.

    filter_name : str, optional
        Filter name to store in the output catalog. If None, the
        magnitude column name is used.

    half_width_arcsec : float, optional
        Half-width of the square field (arcseconds). Default corresponds
        to a 7.5"x7.5" field.

    Returns
    -------
    pandas.DataFrame or bool
        Structured DataFrame containing sources inside the field with columns:

        - 'SourceID'
        - 'RA'
        - 'DEC'
        - 'filter'
        - 'mag'

        Returns False if an error occurs during construction.

    Notes
    -----
    RA selection accounts for cos(dec) correction to preserve angular size.
    """

    HALF_WIDTH_DEG = half_width_arcsec / 3600.0

    # RA correction
    cos_dec = np.cos(np.deg2rad(dec_ref_deg))
    ra_half_width = HALF_WIDTH_DEG / cos_dec
    dec_half_width = HALF_WIDTH_DEG

    nearby_stars = catalog[
        (catalog[ra_colname] >= ra_ref_deg - ra_half_width) &
        (catalog[ra_colname] <= ra_ref_deg + ra_half_width) &
        (catalog[dec_colname] >= dec_ref_deg - dec_half_width) &
        (catalog[dec_colname] <= dec_ref_deg + dec_half_width)
    ]

    # Define output dtype
    out_dtype = np.dtype([
        ('SourceID', 'S8'),
        ('RA', '>f8'),
        ('DEC', '>f8'),
        ('filter', 'S7'),
        ('mag', '>f8')
    ])

    try:

        # Create structured array
        out = np.empty(len(nearby_stars), dtype=out_dtype)

        out['SourceID'] = nearby_stars[id_colname].astype('S8')
        out['RA'] = nearby_stars[ra_colname]
        out['DEC'] = nearby_stars[dec_colname]

        if filter_name is not None:
            out['filter'] = np.array([filter_name] * len(nearby_stars))
        else:
            out['filter'] = np.array([mag_colname.encode()] * len(nearby_stars))
        out['mag'] = nearby_stars[mag_colname]

        nearby_stars_df = pd.DataFrame(out)

    except Exception:
        return False

    return nearby_stars_df


def filter_field(
    field_catalog,
    bright_star_limit,
    dim_mag_limit,
    min_star_number_brighter_than_dim_lim,
    max_star_number_brighter_than_dim_lim,
    mag_colname='mag'
):
    """
    Apply quality criteria to a reference-star-centered field.

    Parameters
    ----------
    field_catalog : pandas.DataFrame
        Catalog of sources within the candidate field.

    bright_star_limit : float
        Field is rejected if any star is brighter than this magnitude.

    dim_mag_limit : float
        Magnitude threshold used to count bright nearby stars.

    min_star_number_brighter_than_dim_lim : int
        Minimum number of stars with mag <= dim_mag_limit required.

    max_star_number_brighter_than_dim_lim : int
        Maximum number of stars with mag <= dim_mag_limit allowed.

    mag_colname : str, optional
        Name of magnitude column (default 'mag').

    Returns
    -------
    bool
        True if the field satisfies all conditions, False otherwise.

    Selection Criteria
    ------------------
    A field is rejected if:

    1. Any star has mag < bright_star_limit.
    2. The number of stars with mag <= dim_mag_limit is outside
       the allowed range.

    Notes
    -----
    Diagnostic messages are printed explaining rejection reasons.
    """

    try:
        # Reject if any nearby star is too bright
        if (field_catalog[mag_colname] < bright_star_limit).any():
            print(f"    Discarded: nearby star with Mag < {bright_star_limit} found")
            return False

        nearby_bright_stars = field_catalog[field_catalog[mag_colname] <= dim_mag_limit]
        n_near_bright = len(nearby_bright_stars)

        # Reject if field is too sparse or too crowded
        if not (min_star_number_brighter_than_dim_lim <= n_near_bright <= max_star_number_brighter_than_dim_lim):
            print("    Discarded: {} nearby stars (outside {}-{} range)".format(
                n_near_bright, min_star_number_brighter_than_dim_lim, max_star_number_brighter_than_dim_lim
            ))
            return False

        print('-- Accepted Field --')
        return True

    except Exception as e:
        print(e)
        return False


def save_field_candidate_csv(
    field_catalog,
    output_dir,
    outname
):
    """
    Save a candidate field catalog to a CSV file.

    Parameters
    ----------
    field_catalog : pandas.DataFrame
        Catalog of sources to save.

    output_dir : str
        Directory where the CSV file will be written.
        Created if it does not exist.

    outname : str
        Name of the output CSV file.

    Returns
    -------
    str
        Full path to the saved CSV file.

    Notes
    -----
    The file is saved using `pandas.DataFrame.to_csv`.
    """

    out_path = os.path.join(output_dir, outname)
    field_catalog.to_csv(out_path)
    return out_path


def plot_field_candidate(
    field_catalog,
    ra_refstar_deg,
    dec_refstar_deg,
    mag_refstar,
    dim_mag_limit,
    output_dir,
    ra_colname='RA',
    dec_colname='DEC',
    mag_colname='mag',
    title_str=None
):
    """
    Generate a sky scatter plot of a reference-star-centered field.

    Parameters
    ----------
    field_catalog : pandas.DataFrame
        Catalog of sources inside the field.

    ra_refstar_deg : float
        Reference star Right Ascension (degrees).

    dec_refstar_deg : float
        Reference star Declination (degrees).

    mag_refstar : float
        Magnitude of the reference star.

    dim_mag_limit : float
        Threshold magnitude used to count bright sources.

    output_dir : str
        Directory where the PNG file will be saved.

    ra_colname, dec_colname : str, optional
        Names of coordinate columns (default 'RA', 'DEC').

    mag_colname : str, optional
        Name of magnitude column (default 'mag').

    title_str : str, optional
        Optional title for the plot.

    Behavior
    --------
    - Marker sizes scale with magnitude (brighter -> bigger).
    - Reference star plotted as a black star marker.
    - RA axis is inverted (astronomical convention).
    - Axes formatted in sexagesimal coordinates.

    Output
    ------
    Saves a PNG file named:

        "{ra}_{dec}_{mag}.png"

    in `output_dir`.
    """

    fig, ax1 = plt.subplots(figsize=(10, 10))

    # --- Size scaling: smaller A_F606W → larger dots ---
    A = field_catalog[mag_colname].values

    # mask valid rows
    mask = np.isfinite(A)

    # if nothing valid, skip this file safely
    if not np.any(mask):
        print(f"Skipping field: no valid {mag_colname} values")
        return

    A_valid = A[mask]

    A_min, A_max = A_valid.min(), A_valid.max()

    # avoid divide-by-zero if constant
    if A_max == A_min:
        sizes = np.full_like(A_valid, 50.0)
    else:
        A_norm = (A_valid - A_min) / (A_max - A_min)
        sizes = 10 + 90 * (1 - A_norm)

    bright_stars = field_catalog[field_catalog[mag_colname] <= dim_mag_limit]

    ax1.scatter(
        field_catalog.loc[mask, ra_colname],
        field_catalog.loc[mask, dec_colname],
        s=sizes,
        alpha=0.7,
        label=f"Field Sources, N={mask.sum()}\nBrighter than mag {dim_mag_limit}={len(bright_stars)}"
    )

    ax1.plot(
        ra_refstar_deg,
        dec_refstar_deg,
        markersize=15,
        marker='*',
        label="Reference star, \n mag = {}".format(mag_refstar),
        color='black'
    )

    # Reverse RA axis (standard astronomical convention)
    ax1.invert_xaxis()

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(ra_formatter))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(dec_formatter))

    ax1.set_xlabel("RA (h:m:s)")
    ax1.set_ylabel("DEC (d:m:s)")
    ax1.legend(loc=1)
    if title_str is not None:
        ax1.set_title(title_str)

    os.makedirs(output_dir, exist_ok=True)

    outname = f"{ra_refstar_deg}_{dec_refstar_deg}_{mag_refstar}.png"
    fig.savefig(
        os.path.join(output_dir, outname),
        dpi=200,
        bbox_inches="tight"
    )
    print(f'Plot saved as: {outname}')
    plt.close()
    return


def generate_finding_charts(
    fits_image_path,
    field_catalog,
    ra_refstar_deg,
    dec_refstar_deg,
    mag_refstar,
    dim_mag_limit,
    output_dir,
    hdu=1,
    ra_colname='RA',
    dec_colname='DEC',
    mag_colname='mag',
    title_str=None
):
    """
    Generate a WCS-based finding chart centered on a reference star.

    Parameters
    ----------
    fits_image_path : str
        Path to the FITS image.

    field_catalog : pandas.DataFrame
        Catalog of sources within the NFM field.

    ra_refstar_deg : float
        Reference star Right Ascension (degrees).

    dec_refstar_deg : float
        Reference star Declination (degrees).

    mag_refstar : float
        Magnitude of the reference star.

    dim_mag_limit : float
        Magnitude threshold for highlighting bright nearby stars.

    output_dir : str
        Directory where the finding chart will be saved.

    hdu : int, optional
        FITS HDU index containing image data (default 1).

    ra_colname, dec_colname : str, optional
        Names of coordinate columns.

    mag_colname : str, optional
        Name of magnitude column.

    title_str : str, optional
        Optional plot title.

    Behavior
    --------
    - Uses WCS projection from FITS header.
    - Displays image using percentile-based clipping.
    - Draws 7.5"x7.5" NFM field rectangle.
    - Marks:
        * Reference star (cyan star)
        * All field sources (yellow stars)
        * Bright sources (red stars)
    - Displays coordinate grid in sky coordinates.

    Output
    ------
    Saves:

        "{ra}_{dec}_{mag}_FC.png"

    in `output_dir`.
    """

    # Load FITS
    with fits.open(fits_image_path) as hdul:
        data = hdul[hdu].data
        header = hdul[hdu].header

    # Build WCS from header
    wcs = WCS(header)

    bright_nearby_stars = field_catalog[field_catalog[mag_colname] <= dim_mag_limit]

    field_coords = SkyCoord(
        field_catalog[ra_colname].values * u.deg, field_catalog[dec_colname].values * u.deg, frame="icrs"
    )
    x_field, y_field = wcs.world_to_pixel(field_coords)

    field_coords_bright = SkyCoord(
        bright_nearby_stars[ra_colname].values * u.deg, bright_nearby_stars[dec_colname].values * u.deg, frame="icrs"
    )
    x_field_bright, y_field_bright = wcs.world_to_pixel(field_coords_bright)

    coord = SkyCoord(ra_refstar_deg * u.deg, dec_refstar_deg * u.deg, frame="icrs")
    x_center, y_center = wcs.world_to_pixel(coord)

    # Half-size = 30 arcsec
    half_size = 3.25 * u.arcsec

    half_box_plot = 5.2 * u.arcsec

    coord_dx_plot = SkyCoord(
        ra=coord.ra + half_box_plot / np.cos(coord.dec),
        dec=coord.dec,
        frame="icrs"
    )

    coord_dy_plot = SkyCoord(
        ra=coord.ra,
        dec=coord.dec + half_box_plot,
        frame="icrs"
    )

    x_dx_plot, _ = wcs.world_to_pixel(coord_dx_plot)
    _, y_dy_plot = wcs.world_to_pixel(coord_dy_plot)

    dx_plot = abs(x_dx_plot - x_center)
    dy_plot = abs(y_dy_plot - y_center)

    # Offset coordinate in RA and Dec
    coord_dx = SkyCoord(
        ra=coord.ra + half_size / np.cos(coord.dec),
        dec=coord.dec,
        frame="icrs"
    )

    coord_dy = SkyCoord(
        ra=coord.ra,
        dec=coord.dec + half_size,
        frame="icrs"
    )

    x_dx, _ = wcs.world_to_pixel(coord_dx)
    _, y_dy = wcs.world_to_pixel(coord_dy)

    # Pixel half-sizes
    dx = abs(x_dx - x_center)
    dy = abs(y_dy - y_center)

    # Clip values (edit these)
    vmin, vmax = np.nanpercentile(data, [0, 99.9])

    # Plot
    fig = plt.figure(figsize=(14, 12))
    ax = plt.subplot(projection=wcs)

    im = ax.imshow(
        data,
        origin="lower",
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )

    # Axis labels
    ax.set_xlabel("Right Ascension (J2000)", fontsize=22)
    ax.set_ylabel("Declination (J2000)", fontsize=22)

    rect = Rectangle(
        (x_center - dx, y_center - dy),  # bottom-left corner
        2 * dx,
        2 * dy,
        edgecolor="red",
        facecolor="none",
        linewidth=2,
        label='NFM Field, 7.5"x7.5"'
    )

    ax.add_patch(rect)

    # Format ticks as sexagesimal
    # ax.coords[0].set_major_formatter('hh:mm:ss')
    # ax.coords[1].set_major_formatter('dd:mm:ss')
    ax.coords[0].set_ticklabel(size=18)  # RA
    ax.coords[1].set_ticklabel(size=18)  # Dec
    ax.coords[0].set_ticks(size=8, width=2)
    ax.coords[1].set_ticks(size=8, width=2)

    ax.plot(
        # x_center,
        # y_center,
        ra_refstar_deg,
        dec_refstar_deg,
        marker='*',
        color='cyan',
        markersize=18,
        markeredgecolor='black',
        markeredgewidth=1.5,
        label=f'RA: {ra_refstar_deg}\nDEC: {dec_refstar_deg}\nMag: {mag_refstar}',
        transform=ax.get_transform('world')
    )

    ax.scatter(
        # x_field,
        # y_field,
        field_catalog[ra_colname].values,
        field_catalog[dec_colname].values,
        marker='*',
        color='yellow',
        transform=ax.get_transform('world')
    )

    ax.scatter(
        # x_field_bright,
        # y_field_bright,
        bright_nearby_stars[ra_colname].values,
        bright_nearby_stars[dec_colname].values,
        marker='*',
        color='red',
        label=r'Mag $\leq$ {}, N = {}'.format(dim_mag_limit, len(bright_nearby_stars)),
        transform=ax.get_transform('world')
    )

    # Optional: grid
    ax.grid(color="white", ls="dotted", alpha=0.5)

    ax.set_xlim(x_center - dx_plot, x_center + dx_plot)
    ax.set_ylim(y_center - dy_plot, y_center + dy_plot)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Flux", fontsize=20)
    cbar.ax.tick_params(labelsize=16, width=2)

    ax.legend(fontsize=15, loc=1)

    if title_str is not None:
        ax.set_title(title_str)

    os.makedirs(output_dir, exist_ok=True)

    outname = "{:.5f}_{:.5f}_{:.5f}_FC.png".format(ra_refstar_deg, dec_refstar_deg, mag_refstar)
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, outname),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()
