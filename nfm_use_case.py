import os
import pandas as pd
import field_finder.nfm_finder as nfm

"""
NFM field search use case for a list of clusters.

For each cluster:
1. Load calibrated catalog and FITS image.
2. Select reference stars within a magnitude range.
3. Build square fields centered on each reference star.
4. Apply selection criteria to identify valid NFM field candidates.
5. Save accepted field catalogs, diagnostic plots, and finding charts.
"""


# -------------------------------
# Paths and column configuration
# -------------------------------

DATA_PATH = os.path.join('nfm_use_case', 'CATALOG_DATA', 'acsggct_data')

RA_COLNAME = 'RA'
DEC_COLNAME = 'Dec'
MAG_COLNAME = 'Vvega'
ID_COLNAME = 'id'
FILTER_NAME = 'A_F606W'

FIELD_CANDIDATES_CATALOG_OUTDIR = 'nfm_use_case/OUTPUTS/nfm_candidates_catalogs'
FIELD_CANDIDATES_PLOTS_OUTDIR = 'nfm_use_case/OUTPUTS/nfm_candidates_plots'
FIELD_CANDIDATES_FINDING_CHARTS_OUTDIR = 'nfm_use_case/OUTPUTS/nfm_candidates_finding_charts'

DIM_MAG_LIMIT = 18

# -----------------------------------------
# Loop over all clusters in DATA_PATH
# -----------------------------------------

for cluster_path in os.listdir(DATA_PATH):

    # Skip hidden/system files
    if cluster_path.startswith('.'):
        continue

    print(cluster_path)

    cluster_name = cluster_path
    print(f'-- Processing Cluster: {cluster_name} --')

    # -----------------------------------------
    # Locate FITS image and calibrated catalog
    # -----------------------------------------

    file_paths = os.listdir(os.path.join(
        DATA_PATH, cluster_name
    ))

    fits_file = next(f for f in file_paths if f.endswith('.fits'))
    catalog_file = next(f for f in file_paths if 'withWCS.cal' in f)

    catalog = pd.read_csv(
        os.path.join(
            DATA_PATH, cluster_name, catalog_file
        ),
        sep="\s+",
        comment='#'
    )

    print(fits_file)
    print(catalog_file)

    # -----------------------------------------
    # Select potential reference stars
    # -----------------------------------------
    refstars = nfm.find_reference_stars(
        catalog=catalog,
        ra_colname=RA_COLNAME,
        dec_colname=DEC_COLNAME,
        mag_colname=MAG_COLNAME,
        refstar_mag_lower_lim=11.9,
        refstar_mag_upper_lim=12.9
    )

    # Prepare output directories for this cluster
    cat_outdir = os.path.join(FIELD_CANDIDATES_CATALOG_OUTDIR, cluster_name)
    plots_outdir = os.path.join(FIELD_CANDIDATES_PLOTS_OUTDIR, cluster_name)
    finding_charts_outdir = os.path.join(FIELD_CANDIDATES_FINDING_CHARTS_OUTDIR, cluster_name)

    # -----------------------------------------
    # For each reference star, search candidate fields
    # -----------------------------------------
    for index, row in refstars.iterrows():

        ra_refstar = row[RA_COLNAME]
        dec_refstar = row[DEC_COLNAME]
        mag_refstar = row[MAG_COLNAME]

        # Build square field centered on reference star
        field_candidate = nfm.find_fields_with_reference_star(
            catalog=catalog,
            ra_ref_deg=row[RA_COLNAME],
            dec_ref_deg=row[DEC_COLNAME],
            ra_colname=RA_COLNAME,
            dec_colname=DEC_COLNAME,
            mag_colname=MAG_COLNAME,
            id_colname=ID_COLNAME,
            half_width_arcsec=7.5/2
            )

        # Apply selection criteria to candidate field
        accepted_candidate = nfm.filter_field(
            field_catalog=field_candidate,
            bright_star_limit=10,
            dim_mag_limit=DIM_MAG_LIMIT,
            min_star_number_brighter_than_dim_lim=40,
            max_star_number_brighter_than_dim_lim=110
        )

        # -----------------------------------------
        # If field passes criteria → save products
        # -----------------------------------------
        if accepted_candidate:

            os.makedirs(cat_outdir, exist_ok=True)

            # Save field catalog
            nfm.save_field_candidate_csv(
                field_catalog=field_candidate,
                output_dir=cat_outdir,
                outname=f'{cluster_name}_{ra_refstar:.5f}_{dec_refstar:5f}_{mag_refstar}.csv'
            )

            # Generate diagnostic plot
            nfm.plot_field_candidate(
                field_catalog=field_candidate,
                ra_refstar_deg=ra_refstar,
                dec_refstar_deg=dec_refstar,
                mag_refstar=mag_refstar,
                dim_mag_limit=DIM_MAG_LIMIT,
                output_dir=plots_outdir,
                title_str=f'{cluster_name}'
            )

            # Generate finding chart using FITS image
            nfm.generate_finding_charts(
                fits_image_path=os.path.join(
                    DATA_PATH, cluster_name, fits_file
                    ),
                hdu=0,
                field_catalog=field_candidate,
                ra_refstar_deg=ra_refstar,
                dec_refstar_deg=dec_refstar,
                mag_refstar=mag_refstar,
                dim_mag_limit=DIM_MAG_LIMIT,
                output_dir=finding_charts_outdir
            )
