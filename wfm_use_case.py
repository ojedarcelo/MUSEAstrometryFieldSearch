import os
import pandas as pd
import field_finder.wfm_finder as wfm
import matplotlib.pyplot as plt
from aux_funcs import query_images_from_legacy_survey

"""
WFM sliding-grid field search for a list of clusters.

For each observable cluster:
1. Load ACS and HSC catalogs.
2. Use HSC outskirts catalog to search candidate WFM fields via sliding grid.
3. Apply density and magnitude constraints.
4. Save accepted field catalogs and diagnostic plots.
5. Download Legacy Survey FITS cutouts and generate finding charts.
"""


plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

# -------------------------------
# Paths and column configuration
# -------------------------------

ACS_DATA_PATH = os.path.join('wfm_use_case', 'CATALOG_DATA', 'acsggct_data')
HSC_DATA_PATH = os.path.join('wfm_use_case', 'CATALOG_DATA', 'hsc_data_outskirts')

OBSERVABLE_CLUSTERS_PATH = os.path.join(
    'wfm_use_case',
    'CATALOG_DATA',
    'all_clusters_acs_observable_february_outskirts.csv'
)

FILENAME_TARGNAME_OBSERVABLE_PATH = os.path.join('wfm_use_case', 'CATALOG_DATA', 'filename_targname_observable.csv')

# Column mappings per catalog
ACS_RA_COLNAME = 'RA'
ACS_DEC_COLNAME = 'Dec'
ACS_MAG_COLNAME = 'Vvega'
ACS_ID_COLNAME = 'id'
ACS_FILTER_NAME = 'A_F606W'

HSC_RA_COLNAME = 'MatchRA'
HSC_DEC_COLNAME = 'MatchDec'
HSC_MAG_COLNAME = 'A_F606W'
HSC_ID_COLNAME = 'MatchID'
HSC_FILTER_NAME = 'A_F606W'

# -------------------------------
# Output directories
# -------------------------------

FIELD_CANDIDATES_CATALOG_OUTDIR = os.path.join(
    'wfm_use_case',
    'OUTPUTS',
    'wfm_candidates_catalogs'
    )
FIELD_CANDIDATES_PLOTS_OUTDIR = os.path.join(
    'wfm_use_case',
    'OUTPUTS',
    'wfm_candidates_plots'
    )

FIELD_CANDIDATES_FITS_OUTDIR = os.path.join(
    'wfm_use_case',
    'OUTPUTS',
    'wfm_candidates_fits_images'
    )

FIELD_CANDIDATES_FINDING_CHARTS_OUTDIR = os.path.join(
    'wfm_use_case',
    'OUTPUTS',
    'wfm_candidates_finding_charts'
    )

DIM_MAG_LIMIT = 20

# -------------------------------
# Load observable cluster metadata
# -------------------------------
filename_targname = pd.read_csv(FILENAME_TARGNAME_OBSERVABLE_PATH)
observable_clusters = pd.read_csv(OBSERVABLE_CLUSTERS_PATH)
observable_set = set(observable_clusters['Cluster'])

# -------------------------------
# Loop over ACS clusters
# -------------------------------
for cluster_path in os.listdir(ACS_DATA_PATH):
    if cluster_path.startswith('.'):
        continue
    print(cluster_path)

    try:
        targname = filename_targname[filename_targname['filename'] == cluster_path]['targetname'].values[0]

        # Skip clusters that are not observable
        if targname not in observable_set:
            continue

    except Exception:
        continue

    cluster_name = cluster_path

    print(f'-- Processing Cluster: {cluster_name} --')

    # -----------------------------------------
    # Load HSC catalog
    # -----------------------------------------
    try:
        hsc_catalog = pd.read_csv(
            os.path.join(
                HSC_DATA_PATH, f'{cluster_name}_hsc_outskirts.csv'
            )
        )

    except Exception:
        continue

    # -----------------------------------------
    # Sliding-grid WFM field search (HSC)
    # -----------------------------------------
    accepted_fields_df, accepted_catalogs = wfm.find_fields_with_sliding_grid(
        catalog=hsc_catalog,
        ra_colname=HSC_RA_COLNAME,
        dec_colname=HSC_DEC_COLNAME,
        mag_colname=HSC_MAG_COLNAME,
        id_colname=HSC_ID_COLNAME,
        dim_mag_limit=DIM_MAG_LIMIT,
        bright_star_limit=10,
        min_star_number_brighter_than_dim_lim=20,
        max_star_number_brighter_than_dim_lim=110,
        min_star_number=250,
        max_star_number=1500,
        grid_step_arcsec=10,
        csv_outdir=os.path.join(
            FIELD_CANDIDATES_CATALOG_OUTDIR,
            cluster_name
            ),
        plots_outdir=os.path.join(
            FIELD_CANDIDATES_PLOTS_OUTDIR,
            cluster_name
            ),
        cluster_name=cluster_name,
    )

    os.makedirs(FIELD_CANDIDATES_FITS_OUTDIR, exist_ok=True)

    # -----------------------------------------
    # For each accepted field:
    #   - download Legacy Survey image
    #   - generate finding chart
    # -----------------------------------------
    for index, row in accepted_fields_df.iterrows():
        ra_c = row['ra']
        dec_c = row['dec']

        # Download FITS cutout
        query_images_from_legacy_survey(
            ra_pointing=ra_c,
            dec_pointing=dec_c,
            cluster_name=cluster_name,
            output_dir=FIELD_CANDIDATES_FITS_OUTDIR
        )

        os.makedirs(FIELD_CANDIDATES_FINDING_CHARTS_OUTDIR, exist_ok=True)

        try:

            wfm.generate_finding_charts(
                fits_image_path='{}/{}_{}_{}.fits'.format(
                    FIELD_CANDIDATES_FITS_OUTDIR,
                    cluster_name,
                    ra_c,
                    dec_c
                ),
                field_catalog=accepted_catalogs[(ra_c, dec_c)],
                ra_colname='RA',
                dec_colname='DEC',
                mag_colname='mag',
                ra_center_deg=ra_c,
                dec_center_deg=dec_c,
                dim_mag_limit=DIM_MAG_LIMIT,
                output_dir=f'{FIELD_CANDIDATES_FINDING_CHARTS_OUTDIR}/{cluster_name}',
                cluster_name=cluster_name
            )

        except Exception as e:
            print()
            print(e)
            print(
                'No image found for RA: {}, DEC: {}'.format(ra_c, dec_c)
            )
            print()
            print()
            continue
