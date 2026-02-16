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
from astropy.visualization import ImageNormalize, PercentileInterval, LogStretch
from aux_funcs import ra_formatter, dec_formatter


plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})


def save_field_candidate_csv(
    field_catalog,
    output_dir,
    outname
):
    """
    Save a field candidate catalog to CSV.

    Parameters
    ----------
    field_catalog : pandas.DataFrame
        Catalog of sources belonging to a candidate field. Expected to
        contain at least RA, DEC, magnitude, and source identifier columns.

    output_dir : str
        Directory where the CSV file will be written. Created if it does not exist.

    outname : str
        Output filename (including extension, e.g. '.csv').

    Notes
    -----
    The function does not return anything. The file is written to:

        os.path.join(output_dir, outname)
    """

    out_path = os.path.join(output_dir, outname)
    field_catalog.to_csv(out_path)
    return


def filter_field(
    field_catalog,
    bright_star_limit,
    dim_mag_limit,
    min_star_number_brighter_than_dim_lim,
    max_star_number_brighter_than_dim_lim,
    min_star_number,
    max_star_number,
    mag_colname='mag'
):
    """
    Apply selection criteria to determine whether a candidate field is acceptable.

    Parameters
    ----------
    field_catalog : pandas.DataFrame
        Catalog of sources inside the candidate field.

    bright_star_limit : float
        Reject the field if any star has magnitude strictly smaller than this value
        (i.e., too bright).

    dim_mag_limit : float
        Magnitude threshold used to count "bright enough" nearby stars.

    min_star_number_brighter_than_dim_lim : int
        Minimum number of stars with magnitude <= dim_mag_limit required.

    max_star_number_brighter_than_dim_lim : int
        Maximum number of stars with magnitude <= dim_mag_limit allowed.

    min_star_number : int
        Minimum total number of stars required in the field.

    max_star_number : int
        Maximum total number of stars allowed in the field.

    mag_colname : str, optional
        Name of the magnitude column. Default is 'mag'.

    Returns
    -------
    bool
        True if the field satisfies all constraints, False otherwise.

    Selection Criteria
    ------------------
    A field is rejected if:

    1. Any star is brighter than `bright_star_limit`.
    2. The number of stars with mag <= dim_mag_limit is outside the allowed range.
    3. The total number of stars is outside the allowed range.

    Notes
    -----
    Diagnostic messages are printed explaining why a field is discarded.
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
            print("    Discarded: {} nearby bright stars (outside {}-{} range)".format(
                n_near_bright, min_star_number_brighter_than_dim_lim, max_star_number_brighter_than_dim_lim
            ))
            return False

        if not (min_star_number <= len(field_catalog) <= max_star_number):
            print("    Discarded: {} total stars (outside {}-{} range)".format(
                len(field_catalog), min_star_number, max_star_number
            ))
            return False

        print('-- Accepted Field --')
        return True

    except Exception as e:
        print(e)
        return False


def plot_field_candidate(
    field_catalog,
    ra_colname,
    dec_colname,
    mag_colname,
    ra_center_deg,
    dec_center_deg,
    dim_mag_limit,
    output_dir,
    title_str=None
):
    """
    Create and save a sky scatter plot of a candidate field.

    Parameters
    ----------
    field_catalog : pandas.DataFrame
        Catalog of sources in the candidate field.

    ra_colname : str
        Name of the RA column (degrees).

    dec_colname : str
        Name of the DEC column (degrees).

    mag_colname : str
        Name of the magnitude column.

    ra_center_deg : float
        Right Ascension of the field center in degrees.

    dec_center_deg : float
        Declination of the field center in degrees.

    dim_mag_limit : float
        Magnitude threshold used to count bright stars in the legend.

    output_dir : str
        Directory where the PNG plot will be saved.

    title_str : str, optional
        Optional title for the plot.

    Behavior
    --------
    - Stars brighter than mag 21 are plotted.
    - Marker sizes scale inversely with magnitude (brighter → larger).
    - The field center is marked in black.
    - RA axis is inverted (astronomical convention).
    - Axes are formatted in sexagesimal (hh:mm:ss, dd:mm:ss).

    Output
    ------
    A PNG file named:

        "{ra_center_deg}_{dec_center_deg}.png"

    is saved to `output_dir`.
    """

    fig, ax1 = plt.subplots(figsize=(10, 10))

    bright_enough_stars = field_catalog[field_catalog[mag_colname] <= 21]

    # --- Size scaling: smaller A_F606W → larger dots ---
    A = bright_enough_stars[mag_colname].values

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
        bright_enough_stars.loc[mask, ra_colname],
        bright_enough_stars.loc[mask, dec_colname],
        s=sizes,
        alpha=0.7,
        label=f"Field Sources, Brighter than mag 21={mask.sum()}\nBrighter than mag {dim_mag_limit}={len(bright_stars)}"
    )

    ax1.scatter(
        ra_center_deg,
        dec_center_deg,
        s=5,
        label="Field center",
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

    outname = f"{ra_center_deg}_{dec_center_deg}.png"
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
    ra_colname,
    dec_colname,
    mag_colname,
    ra_center_deg,
    dec_center_deg,
    dim_mag_limit,
    output_dir,
    cluster_name,
    percentage=95,
    title_str=None,
    hdu=0
):
    """
    Generate and save a finding chart for an accepted field using a FITS image.

    Parameters
    ----------
    fits_image_path : str
        Path to the FITS image file.

    field_catalog : pandas.DataFrame
        Catalog of sources in the field.

    ra_colname : str
        Name of the RA column (degrees).

    dec_colname : str
        Name of the DEC column (degrees).

    mag_colname : str
        Name of the magnitude column.

    ra_center_deg : float
        Field center Right Ascension in degrees.

    dec_center_deg : float
        Field center Declination in degrees.

    dim_mag_limit : float
        Magnitude threshold used to highlight bright stars.

    output_dir : str
        Directory where the finding chart will be saved.

    cluster_name : str
        Name of the cluster (used in output filename).

    percentage : float, optional
        Percentile interval used for image normalization (default 95).

    title_str : str, optional
        Optional title for the plot.

    hdu : int, optional
        FITS HDU index to use (default 0).

    Behavior
    --------
    - Uses WCS projection from FITS header.
    - Applies percentile-based normalization with logarithmic stretch.
    - Draws a 1'x1' red rectangle centered on the field.
    - Marks:
        * Field center (cyan star)
        * All catalog stars (yellow stars)
        * Stars with mag <= dim_mag_limit (red stars)
    - Displays coordinate grid in sexagesimal.

    Output
    ------
    Saves:

        "{cluster_name}_{ra}_{dec}_FC.png"

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

    coord = SkyCoord(ra_center_deg * u.deg, dec_center_deg * u.deg, frame="icrs")
    x_center, y_center = wcs.world_to_pixel(coord)

    # Half-size = 30 arcsec
    half_size = 30 * u.arcsec

    half_box_plot = 35 * u.arcsec

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

    interval = PercentileInterval(percentage)   # adjust to match CARTA slider
    vmin, vmax = interval.get_limits(data)

    norm = ImageNormalize(
        vmin=vmin,
        vmax=vmax,
        stretch=LogStretch()
    )

    # Plot
    fig = plt.figure(figsize=(14, 12))
    ax = plt.subplot(projection=wcs)

    im = ax.imshow(
        data,
        origin="lower",
        cmap="gray",
        norm=norm
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
        label="WFM Field, 1'x1'"
    )

    ax.add_patch(rect)

    # Format ticks as sexagesimal
    ax.coords[0].set_major_formatter('hh:mm:ss')
    ax.coords[1].set_major_formatter('dd:mm:ss')
    ax.coords[0].set_ticklabel(size=18)  # RA
    ax.coords[1].set_ticklabel(size=18)  # Dec
    ax.coords[0].set_ticks(size=8, width=2)
    ax.coords[1].set_ticks(size=8, width=2)

    ax.plot(
        x_center,
        y_center,
        marker='*',
        color='cyan',
        markersize=15,
        markeredgecolor='black',
        markeredgewidth=1.5,
        label=f'RA: {ra_center_deg}\nDEC: {dec_center_deg}'
    )

    ax.scatter(
        x_field,
        y_field,
        marker='*',
        color='yellow'
    )

    ax.scatter(
        x_field_bright,
        y_field_bright,
        marker='*',
        color='red',
        label=r'Mag $\leq$ {}, N = {}'.format(dim_mag_limit, len(bright_nearby_stars))
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

    outname = "{}_{:.5f}_{:.5f}_FC.png".format(cluster_name, ra_center_deg, dec_center_deg)
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, outname),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()


def find_fields_with_sliding_grid(
    catalog,
    ra_colname,
    dec_colname,
    mag_colname,
    id_colname,
    dim_mag_limit,
    bright_star_limit,
    min_star_number_brighter_than_dim_lim,
    max_star_number_brighter_than_dim_lim,
    min_star_number,
    max_star_number,
    grid_step_arcsec,
    csv_outdir,
    plots_outdir,
    cluster_name=None,
    half_width_arcsec=60/2,
    visualize=True,
    visualize_pause=0.000000000000005
):
    """
    Search for acceptable fields using a sliding rectangular kernel.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Full source catalog.

    ra_colname, dec_colname : str
        Names of RA and DEC columns (degrees).

    mag_colname : str
        Name of magnitude column.

    id_colname : str
        Name of source ID column.

    dim_mag_limit : float
        Magnitude threshold used for bright-star counting.

    bright_star_limit : float
        Reject field if any star is brighter than this value.

    min_star_number_brighter_than_dim_lim, max_star_number_brighter_than_dim_lim : int
        Allowed range of stars with mag <= dim_mag_limit.

    min_star_number, max_star_number : int
        Allowed range of total stars in field.

    grid_step_arcsec : float
        Step size of sliding grid in arcseconds.

    csv_outdir : str
        Directory where accepted field catalogs are saved.

    plots_outdir : str
        Directory where field plots are saved.

    cluster_name : str, optional
        Name used in output files and visualizer title.

    half_width_arcsec : float, optional
        Half-width of square field in arcseconds (default 30").

    visualize : bool, optional
        If True, shows interactive sliding visualization.

    visualize_pause : float, optional
        Pause time between frames in visualization.

    Returns
    -------
    accepted_fields_df : pandas.DataFrame
        DataFrame with columns ['ra', 'dec'] for accepted centers.

    accepted_catalogs : dict
        Dictionary mapping (ra, dec) → field catalog DataFrame.

    Notes
    -----
    For each grid center:
        1. Select sources within a rectangular box.
        2. Apply `filter_field`.
        3. Save accepted catalogs and plots.
    """

    HALF_WIDTH_DEG = half_width_arcsec / 3600.0
    GRID_STEP_DEG = grid_step_arcsec / 3600.0

    ra_min, ra_max = catalog[ra_colname].min(), catalog[ra_colname].max()
    dec_min, dec_max = catalog[dec_colname].min(), catalog[dec_colname].max()

    ra_centers = np.arange(ra_min, ra_max, GRID_STEP_DEG)
    dec_centers = np.arange(dec_min, dec_max, GRID_STEP_DEG)

    # ---- NEW containers ----
    accepted_centers = []
    accepted_catalogs = {}

    # Define output dtype
    out_dtype = np.dtype([
        ('SourceID', 'S8'),
        ('RA', '>f8'),
        ('DEC', '>f8'),
        ('filter', 'S7'),
        ('mag', '>f8')
    ])

    visualizer = None
    if visualize:
        visualizer = SlidingGridVisualizer(
            catalog=catalog,
            ra_colname=ra_colname,
            dec_colname=dec_colname,
            mag_colname=mag_colname,
            dim_mag_limit=dim_mag_limit,
            half_width_arcsec=half_width_arcsec,
            pause_time=visualize_pause,
            title=cluster_name
        )

    # =================================================
    # Sliding kernel
    # =================================================
    for ra_c in ra_centers:
        for dec_c in dec_centers:

            cos_dec = np.cos(np.deg2rad(dec_c))
            ra_half_width = HALF_WIDTH_DEG / cos_dec
            dec_half_width = HALF_WIDTH_DEG

            field_candidate = catalog[
                (catalog[ra_colname] >= ra_c - ra_half_width) &
                (catalog[ra_colname] <= ra_c + ra_half_width) &
                (catalog[dec_colname] >= dec_c - dec_half_width) &
                (catalog[dec_colname] <= dec_c + dec_half_width)
            ]

            try:
                out = np.empty(len(field_candidate), dtype=out_dtype)

                out['SourceID'] = field_candidate[id_colname].astype('S8')
                out['RA'] = field_candidate[ra_colname]
                out['DEC'] = field_candidate[dec_colname]
                out['filter'] = np.array([mag_colname.encode()] * len(field_candidate))
                out['mag'] = field_candidate[mag_colname]

                field_candidate_df = pd.DataFrame(out)

            except Exception:
                continue

            accepted_field = filter_field(
                field_catalog=field_candidate_df,
                bright_star_limit=bright_star_limit,
                dim_mag_limit=dim_mag_limit,
                min_star_number_brighter_than_dim_lim=min_star_number_brighter_than_dim_lim,
                max_star_number_brighter_than_dim_lim=max_star_number_brighter_than_dim_lim,
                min_star_number=min_star_number,
                max_star_number=max_star_number,
            )

            if visualizer is not None:
                visualizer.update(ra_c, dec_c, accepted=bool(accepted_field))

            if accepted_field:

                # ---- SAVE ACCEPTED CENTER ----
                accepted_centers.append((ra_c, dec_c))

                # ---- SAVE ACCEPTED CATALOG ----
                accepted_catalogs[(ra_c, dec_c)] = field_candidate_df.copy()

                # ---------- CSV ----------
                os.makedirs(csv_outdir, exist_ok=True)

                save_field_candidate_csv(
                    field_candidate_df,
                    output_dir=csv_outdir,
                    outname='{}_{:.5f}_{:.5f}.csv'.format(
                        cluster_name, ra_c, dec_c
                    )
                )

                # ---------- PLOTS ----------
                os.makedirs(plots_outdir, exist_ok=True)

                plot_field_candidate(
                    field_catalog=field_candidate_df,
                    ra_colname='RA',
                    dec_colname='DEC',
                    mag_colname='mag',
                    ra_center_deg=ra_c,
                    dec_center_deg=dec_c,
                    dim_mag_limit=dim_mag_limit,
                    output_dir=plots_outdir,
                    title_str=cluster_name
                )

    # ---- Convert centers to dataframe ----
    accepted_fields_df = pd.DataFrame(
        accepted_centers,
        columns=['ra', 'dec']
    )

    if visualizer is not None:
        visualizer.close()

    return accepted_fields_df, accepted_catalogs


class SlidingGridVisualizer:
    """
    Interactive visualizer for sliding field selection.

    Displays:
        - Full catalog in gray.
        - Moving rectangular field.
        - Live statistics in title.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Full source catalog.

    ra_colname, dec_colname : str
        Names of coordinate columns.

    mag_colname : str
        Magnitude column name.

    dim_mag_limit : float
        Threshold for counting bright stars.

    half_width_arcsec : float
        Half-width of field in arcseconds.

    pause_time : float or None
        Pause duration between updates.

    title : str or None
        Base title for visualization.
    """

    def __init__(
        self,
        catalog,
        ra_colname,
        dec_colname,
        mag_colname,
        dim_mag_limit,
        half_width_arcsec,
        pause_time=None,
        title=None
    ):
        self.catalog = catalog
        self.ra_colname = ra_colname
        self.dec_colname = dec_colname
        self.mag_colname = mag_colname
        self.dim_mag_limit = dim_mag_limit
        self.pause_time = pause_time

        self.half_width_deg = half_width_arcsec / 3600.0
        self.base_title = title if title else ""

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        # Plot catalog once
        self.ax.scatter(
            catalog[ra_colname],
            catalog[dec_colname],
            s=3,
            alpha=0.5,
            color="gray"
        )

        self.ax.invert_xaxis()
        self.ax.set_xlabel("RA (deg)")
        self.ax.set_ylabel("DEC (deg)")

        # Moving rectangle
        self.rect = Rectangle(
            (0, 0),
            0,
            0,
            edgecolor="red",
            facecolor="none",
            linewidth=2
        )

        self.ax.add_patch(self.rect)
        plt.show(block=False)

    def update(self, ra_center, dec_center, accepted=False):
        """
        Update rectangle position and statistics.

        Parameters
        ----------
        ra_center : float
            Current RA center (degrees).

        dec_center : float
            Current DEC center (degrees).

        accepted : bool, optional
            If True, rectangle is drawn in green; otherwise red.

        Behavior
        --------
        - Moves rectangle.
        - Updates star counts in title.
        - Redraws figure interactively.
        """

        cos_dec = np.cos(np.deg2rad(dec_center))
        ra_half = self.half_width_deg / cos_dec
        dec_half = self.half_width_deg

        # ----- Select stars inside field -----
        field = self.catalog[
            (self.catalog[self.ra_colname] >= ra_center - ra_half) &
            (self.catalog[self.ra_colname] <= ra_center + ra_half) &
            (self.catalog[self.dec_colname] >= dec_center - dec_half) &
            (self.catalog[self.dec_colname] <= dec_center + dec_half)
        ]

        n_total = len(field)
        n_bright = np.sum(field[self.mag_colname] <= self.dim_mag_limit)

        # ----- Update rectangle position -----
        self.rect.set_xy((ra_center - ra_half, dec_center - dec_half))
        self.rect.set_width(2 * ra_half)
        self.rect.set_height(2 * dec_half)

        # ----- Change color depending on acceptance -----
        if accepted:
            self.rect.set_edgecolor("green")
        else:
            self.rect.set_edgecolor("red")

        # ----- Update title -----
        title = (
            f"{self.base_title}\n"
            f"RA={ra_center:.5f}, DEC={dec_center:.5f}\n"
            f"N stars = {n_total}, N mag ≤ {self.dim_mag_limit} = {n_bright}"
        )
        self.ax.set_title(title)

        self.fig.canvas.draw_idle()

        if self.pause_time is not None:
            plt.pause(self.pause_time)

    def close(self):
        """
        Close the interactive visualization window and disable interactive mode.
        """

        plt.ioff()
        plt.close(self.fig)
