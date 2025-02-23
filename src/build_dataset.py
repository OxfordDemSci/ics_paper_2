import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    filename='../logging/logfile.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

pd.set_option('future.no_silent_downcasting', True)

def format_ids(df):
    """
    Formats the institution IDs within a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing institution ID columns.

    Returns:
        DataFrame: The DataFrame with formatted institution IDs.

    The function renames institution ID columns for consistency, removes rows with
    missing IDs, and formats the IDs as integers. It also combines 'Unit of
    assessment number' and 'Multiple submission letter' into a single 'uoa_id'.
    """
    if 'Institution UKPRN code' in df.columns:
        df = df.rename(columns={'Institution UKPRN code': 'inst_id'})
    if 'Institution code (UKPRN)' in df.columns:
        df = df.rename(columns={'Institution code (UKPRN)': 'inst_id'})
    df = df[df['inst_id'] != ' ']

    df = df.astype({'inst_id': 'int'})
    df['uoa_id'] = df['Unit of assessment number'].astype(int).astype(
        str) + df['Multiple submission letter'].fillna('').astype(str)
    return df


def merge_ins_uoa(df1, df2, id1='inst_id', id2='uoa_id'):
    """
    Merges two DataFrames based on institution and unit of assessment IDs.

    Args:
        df1 (pd.DataFrame): The first DataFrame to merge.
        df2 (pd.DataFrame): The second DataFrame to merge.

    Returns:
        DataFrame: The merged DataFrame.

    The function performs a left merge of df2 on df1 based on 'inst_id' and
    'uoa_id'. It asserts that 'inst_id' and 'uoa_id' in df1 are present in df2.
    """
    assert all(df1[id1].isin(df2[id1]))
    assert all(df1[id2].isin(df2[id2]))

    return df1.merge(df2, how='left', on=[id1, id2])


def build_dataset():
    df = pd.read_spss('../data/raw/survey/results_25_06.sav')
    logging.info(f'OK, so weve got {len(df)} people to begin with.')
    logging.info(f"{len(df[(df['Q1'] == 'No') | (df['Q2'] == 'No')])} people did not read PIS/agree to analysis")
    df = df[(df['Q1'] != 'No') & (df['Q2'] != 'No')]
    logging.info(f'This leaves us with {len(df)} rows of observations')
    logging.info(f"There are {len(df[df['Q10']==''])} obs missing for Q10: important.")
    df = df[df['Q10'] != '']
    logging.info(f'We drop these. This leaves us with {len(df)} people.')
    df_clean = pd.DataFrame()
    df['Q3'] = ''
    for q in range(1, 7):
        col_name = 'Q3_' + str(q)
        df[col_name] = df[col_name].astype(str)
        df[col_name] = df[col_name].replace('nan', '')
        df[col_name] = df[col_name].str.strip().str.replace(r':.*', '', regex=True)
        df['Q3'] = df['Q3'] + df[col_name] + ':'
    df['Q3'] = df['Q3'].str.replace(':+', ':', regex=True).str.strip(':')
    df_clean['Q1'] = df['Q3']
    df['Q3_b'] = ''
    for q in range(1, 13):
        col_name = 'Q3_b_' + str(q)
        df[col_name] = df[col_name].astype(str)
        df[col_name] = df[col_name].replace('nan', '')
        df[col_name] = df[col_name].str.strip().str.replace(r':.*', '', regex=True)
        df['Q3_b'] = df['Q3_b'] + df[col_name] + ':'
    df['Q3_b'] = df['Q3_b'].str.replace(':+', ':', regex=True).str.strip(':')
    df_clean['Q2'] = df['Q3_b']
    df_clean['Q3'] = df['Q3_c']
    logging.info(f"There are {len(df_clean['Q3'].unique())} unique institutions")
    orgs = df_clean[df_clean['Q3']!='Other']['Q3'].value_counts().reset_index()
    modal = orgs.sort_values(by='count', ascending=False)['Q3'][0]
    logging.info(f"The modal institution (excluding 'Other') is: {modal}")
    logging.info('-'*20)
    logging.info(df_clean['Q3'].value_counts().to_string())
    df_clean['Q4'] = df['Q4']
    min_phd = df_clean['Q4'].value_counts().reset_index()['Q4'].astype(int).min()
    max_phd = df_clean['Q4'].value_counts().reset_index()['Q4'].astype(int).max()
    mean_phd = df_clean['Q4'].value_counts().reset_index()['Q4'].astype(int).mean()
    logging.info(f'Most recent year of PhD: {max_phd}')
    logging.info(f'Earliest year of PhD: {min_phd}')
    logging.info(f'Mean year of PhD: {int(mean_phd)}')
    df_clean['Q4'] = df_clean['Q4'].fillna(np.nan)
    df_clean['Q4'] = df_clean['Q4'].astype('Int64')
    df_clean['Q4_years'] = 2021 - df_clean['Q4']
    df_clean['Q4_stage'] = np.nan
    df_clean['Q4_stage'] = np.where(df_clean['Q4_years'] <= 10, 'Early', df_clean['Q4_stage'])
    df_clean['Q4_stage'] = np.where((df_clean['Q4_years'] > 10) & (df_clean['Q4_years'] <= 25), 'Middle', df_clean['Q4_stage'])
    df_clean['Q4_stage'] = np.where(df_clean['Q4_years'] > 25, 'Senior', df_clean['Q4_stage'])
    df_clean['Q5'] = df['Q5']
    df_clean['Q5_binary'] = np.nan
    df_clean['Q5_binary'] = np.where(df['Q5'] == 'Female', 1, df_clean['Q5_binary'])
    df_clean['Q5_binary'] = np.where(df['Q5'] == 'Male', 0, df_clean['Q5_binary'])
    df_clean['Q6'] = df['Q6']  # This should probably have a 'prefer not to say'?
    logging.info('-'*20)
    logging.info(df_clean['Q6'].value_counts().to_string())
    logging.info('-'*20)
    df['Q7'] = ''
    for q in range(1, 36):
        col_name = 'Q7_' + str(q)
        df[col_name] = df[col_name].astype(str)
        df[col_name] = df[col_name].replace('nan', '')
        df[col_name] = df[col_name].str.strip().str.replace(r':.*', '', regex=True)
        df['Q7'] = df['Q7'] + df[col_name] + ':'
    df['Q7'] = df['Q7'].str.replace(':+', ':', regex=True).str.strip(':')
    df_clean['Q7'] = df['Q7']
    logging.info('-'*20)
    logging.info(df_clean['Q7'].head(5).to_string())
    logging.info('-'*20)
    extracted = df_clean['Q7'].str.extract(r'([-]?\d+)', expand=False)
    df_clean['Q7_uoa'] = pd.to_numeric(extracted, errors='coerce').astype('Int64')
    logging.info(df_clean['Q7_uoa'].value_counts().sort_index().to_string())
    logging.info('-'*20)
    df_clean['Q8'] = df['Q8']
    logging.info(df_clean['Q8'].value_counts().to_string())
    logging.info('-'*20)
    df_clean['Q8_a'] = df['Q8_a']
    logging.info(f"Unique Q8a: {df['Q8_a'].unique()}")
    logging.info('-'*20)
    df_clean['Q8_uoa'] = df_clean['Q8']
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Clinical Medicine', 1, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Public Health, Health Services and Primary Care', 2, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Allied Health Professions, Dentistry, Nursing and Pharmacy', 3, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Psychology, Psychiatry and Neuroscience', 4, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Biological Sciences', 5, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Agriculture, Food and Veterinary Sciences', 6, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Earth Systems and Environmental Sciences', 7, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Chemistry', 8, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Physics', 9, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Mathematical Sciences', 10, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Computer Science and Informatics', 11, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Engineering', 12, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Architecture, Built Environment and Planning', 13, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Geography and Environmental Studies', 14, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Archaeology', 15, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Economics and Econometrics', 16, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Business and Management Studies', 17, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Law', 18, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Politics and International Studies', 19, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Social Work and Social Policy', 20, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Sociology', 21, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Anthropology and Development Studies', 22, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Education', 23, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Sport and Exercise Sciences, Leisure and Tourism', 24, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Area Studies', 25, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Modern Languages and Linguistics', 26, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'English Language and Literature', 27, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'History', 28, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Classics', 29, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Philosophy', 30, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Theology and Religious Studies', 31, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Art and Design: History, Practice and Theory', 32, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Music, Drama, Dance, Performing Arts, Film and Screen Studies', 33, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Communication, Cultural and Media Studies, Library and Information Management', 34, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8'] == 'Other', np.nan, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = df_clean['Q8_uoa'].astype('Int64')
    logging.info(df_clean['Q8_uoa'].value_counts().to_string())
    logging.info('-'*20)
    logging.info(f"There are {len(df_clean[df_clean['Q8_uoa'].isnull()])} people who put 8_uoa as 'Other' but fit into a UOA")
    df_clean['Q8_uoa'] = np.where(df_clean['Q8_a'] == 'Demography', 21, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8_a'] == 'Medical ethics', 30, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8_a'] == 'Criminology', 21, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8_a'] == 'Political economy', 16, df_clean['Q8_uoa'])
    df_clean['Q8_uoa'] = np.where(df_clean['Q8_a'] == 'Gerontology', 21, df_clean['Q8_uoa'])
    df_clean['Q8_panel'] = np.nan
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 1, 'A', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 2, 'A', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 3, 'A', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 4, 'A', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 5, 'A', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 6, 'A', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 7, 'B', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 8, 'B', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 9, 'B', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 10, 'B', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 11, 'B', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 12, 'B', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 13, 'C', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 14, 'C', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 15, 'C', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 16, 'C', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 17, 'C', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 18, 'C', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 19, 'C', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 20, 'C', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 21, 'C', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 22, 'C', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 23, 'C', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 24, 'C', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 25, 'D', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 26, 'D', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 27, 'D', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 28, 'D', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 29, 'D', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 30, 'D', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 31, 'D', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 32, 'D', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 33, 'D', df_clean['Q8_panel'])
    df_clean['Q8_panel'] = np.where(df_clean['Q8_uoa'] == 34, 'D', df_clean['Q8_panel'])
    df_clean['Q8_stemshape'] = np.nan
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 1, 'STEM', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 2, 'STEM', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 3, 'STEM', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 4, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 5, 'STEM', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 6, 'STEM', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 7, 'STEM', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 8, 'STEM', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 9, 'STEM', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 10, 'STEM', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 11, 'STEM', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 12, 'STEM', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 13, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 14, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 15, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 16, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 17, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 18, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 19, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 20, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 21, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 22, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 23, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 24, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 25, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 26, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 27, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 28, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 29, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 30, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 31, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 32, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 33, 'SHAPE', df_clean['Q8_stemshape'])
    df_clean['Q8_stemshape'] = np.where(df_clean['Q8_uoa'] == 34, 'SHAPE', df_clean['Q8_stemshape'])


    df_clean['Q8_stemshape_binary'] = np.nan
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 1, 1, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 2, 1, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 3, 1, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 4, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 5, 1, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 6, 1, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 7, 1, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 8, 1, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 9, 1, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 10, 1, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 11, 1, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 12, 1, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 13, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 14, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 15, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 16, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 17, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 18, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 19, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 20, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 21, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 22, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 23, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 24, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 25, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 26, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 27, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 28, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 29, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 30, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 31, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 32, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 33, 0, df_clean['Q8_stemshape_binary'])
    df_clean['Q8_stemshape_binary'] = np.where(df_clean['Q8_uoa'] == 34, 0, df_clean['Q8_stemshape_binary'])


    df_clean['Q9'] = df['Q9']
    logging.info('-'*20)
    logging.info(df_clean['Q9'].head(5).to_string())
    logging.info('-'*20)
    df_clean['Q10'] = df['Q10']
    min_weight = df_clean['Q10'].value_counts().reset_index()['Q10'].astype(int).min()
    max_weight = df_clean['Q10'].value_counts().reset_index()['Q10'].astype(int).max()
    mean_weight = df_clean['Q10'].value_counts().reset_index()['Q10'].astype(int).mean()
    logging.info(f'Highest weight for ICS: {max_weight}')
    logging.info(f'Smallest weight for ICS: {min_weight}')
    logging.info(f'Mean weight for ICS: {int(mean_weight)}')
    df_clean['Q10a'] = df['Q11']
    logging.info('-'*20)
    logging.info(df_clean['Q10a'].head(5).to_string())
    logging.info('-' * 20)
    for number in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
        Q = 'Q11_' + number
        df_clean[Q] = df['Q12_' + number + '_a'].astype('Int64')
        logging.info(f"Mean rank for {Q} is {np.round(df_clean[df_clean[Q].notnull()][Q].mean(), 3)}")
    logging.info('-' * 20)
    for q in range(1, 24):
        Q = 'Q12_' + str(q)
        df_clean[Q] = df[['Q13_' + str(q) + '_1', 'Q13_' + str(q) + '_2',
                          'Q13_' + str(q) + '_3', 'Q13_' + str(q) + '_4',
                          'Q13_' + str(q) + '_5', 'Q13_' + str(q) + '_6',
                          'Q13_' + str(q) + '_7']].bfill(axis=1).iloc[:, 0]
        df_clean[Q] = df_clean[Q].str.replace('(3) extremely possible', '3')
        df_clean[Q] = df_clean[Q].str.replace('(-3) not at all possible', '-3')
        df_clean[Q] = df_clean[Q].str.replace('(0) neither', '0')
        df_clean[Q] = df_clean[Q].astype('Int64')
        logging.info(f"Mean rank for {Q} is {np.round(df_clean[df_clean[Q].notnull()][Q].mean(), 3)}")
    df['Q14'] = ''
    for q in range(1, 24):
        col_name = 'Q14_' + str(q)
        df[col_name] = df[col_name].astype(str)
        df[col_name] = df[col_name].replace('nan', '')
        df[col_name] = df[col_name].str.strip().str.replace(r':.*', '', regex=True)
        df['Q14'] = df['Q14'] + df[col_name] + ':'

    df['Q14'] = df['Q14'].str.replace(':+', ':', regex=True).str.strip(':')
    df_clean['12a'] = df['Q14']
    df['Q15'] = ''
    for q in range(1, 24):
        col_name = 'Q15_' + str(q)
        df[col_name] = df[col_name].astype(str)
        df[col_name] = df[col_name].replace('nan', '')
        df[col_name] = df[col_name].str.strip().str.replace(r':.*', '', regex=True)
        df['Q15'] = df['Q15'] + df[col_name] + ':'
    df['Q15'] = df['Q15'].str.replace(':+', ':', regex=True).str.strip(':')
    df_clean['12b'] = df['Q15']
    logging.info('-' * 20)
    df_clean['Q13'] = df['Q16']
    logging.info(df_clean['Q13'].head(5).to_string())
    logging.info('-' * 20)
    df_clean['Q14'] = df['Q17']
    logging.info(df_clean['Q14'].head(5).to_string())
    df['Q18'] = df['Q18'].str.replace('(3) Strongly Agree', '3')
    df['Q18'] = df['Q18'].str.replace('(-3) Strongly Disagree', '-3')
    df['Q18'] = df['Q18'].str.replace('(0) neither', '0')
    df_clean['Q15'] = df['Q18'].astype('Int64')
    logging.info('-' * 20)
    logging.info(f"Q15 mean: {df_clean['Q15'].mean()}")
    df_clean['Q16'] = pd.to_numeric(df['Q19'], errors='coerce').astype('Int64')
    logging.info(f"Q16 mean: {df_clean['Q16'].mean()}")
    logging.info('-' * 20)
    for q in range(1, 11):
        Q = 'Q17_' + str(q)
        df_clean[Q] = df[['Q20_' + str(q) + '_1', 'Q20_' + str(q) + '_2',
                          'Q20_' + str(q) + '_3', 'Q20_' + str(q) + '_4',
                          'Q20_' + str(q) + '_5', 'Q20_' + str(q) + '_6',
                          'Q20_' + str(q) + '_7']].bfill(axis=1).iloc[:, 0]
        df_clean[Q] = df_clean[Q].str.replace('(3) extremely important', '3')
        df_clean[Q] = df_clean[Q].str.replace('(-3) not at all important', '-3')
        df_clean[Q] = df_clean[Q].str.replace('(0) modest importance', '0')
        df_clean[Q] = df_clean[Q].astype('Int64')
        logging.info(f"Mean rank for {Q} is {np.round(df_clean[df_clean[Q].notnull()][Q].mean(), 3)}")
    df_clean['Q17a'] = df['Q21']
    logging.info('-' * 20)
    logging.info(df_clean['Q17a'].head(5).to_string())
    logging.info('-' * 20)
    for q in range(1, 11):
        Q = 'Q18_' + str(q)
        df_clean[Q] = df[['Q22_' + str(q) + '_1', 'Q22_' + str(q) + '_2',
                          'Q22_' + str(q) + '_3', 'Q22_' + str(q) + '_4',
                          'Q22_' + str(q) + '_5', 'Q22_' + str(q) + '_6',
                          'Q22_' + str(q) + '_7']].bfill(axis=1).iloc[:, 0]
        df_clean[Q] = df_clean[Q].str.replace('(3) extremely important', '3')
        df_clean[Q] = df_clean[Q].str.replace('(-3) not at all important', '-3')
        df_clean[Q] = df_clean[Q].str.replace('(0) modestly important', '0')
        df_clean[Q] = df_clean[Q].astype('Int64')
        logging.info(f"Mean rank for {Q} is {np.round(df_clean[df_clean[Q].notnull()][Q].mean(), 3)}")
    logging.info('-' * 20)
    for q in range(1, 11):
        Q = 'Q18a_' + str(q)
        df_clean[Q] = df[['Q23_' + str(q) + '_1', 'Q23_' + str(q) + '_2',
                          'Q23_' + str(q) + '_3', 'Q23_' + str(q) + '_4',
                          'Q23_' + str(q) + '_5', 'Q23_' + str(q) + '_6',
                          'Q23_' + str(q) + '_7']].bfill(axis=1).iloc[:, 0]
        df_clean[Q] = df_clean[Q].str.replace('(3) extremely able', '3')
        df_clean[Q] = df_clean[Q].str.replace('(-3) not at all able', '-3')
        df_clean[Q] = df_clean[Q].str.replace('(0) modestly able', '0')
        df_clean[Q] = df_clean[Q].astype('Int64')
        logging.info(f"Mean rank for {Q} is {np.round(df_clean[df_clean[Q].notnull()][Q].mean(), 3)}")
    df_clean['Q18b'] = df['Q24']
    logging.info('-' * 20)
    logging.info(df_clean['Q18b'].head(5).to_string())
    df_clean['PIPD'] = df['Q28'].astype('Int64')

    raw_results = pd.read_excel('../data/raw/ref/raw_ref_results_data.xlsx', skiprows=6)
    raw_results = format_ids(raw_results)
    raw_results = raw_results.rename(
        columns={'FTE of submitted staff': 'fte',
                 '% of eligible staff submitted': 'fte_pc'})

    ## Make wide score card by institution and uoa_id
    score_types = ['4*', '3*', '2*', '1*', 'Unclassified']  # types of scores
    wide_score_card = pd.pivot(
        raw_results[['inst_id', 'uoa_id', 'Profile'] + score_types],
        index=['inst_id', 'uoa_id'], columns=['Profile'], values=score_types)
    wide_score_card.columns = wide_score_card.columns.map('_'.join)
    wide_score_card = wide_score_card.reset_index()

    # Process environmental data
    raw_env_path = '../data/raw/ref/raw_ref_environment_data.xlsx'
    # Doctoral data
    raw_env_doctoral = pd.read_excel(raw_env_path, sheet_name="ResearchDoctoralDegreesAwarded", skiprows=4)
    raw_env_doctoral = format_ids(raw_env_doctoral)
    number_cols = [col for col in raw_env_doctoral.columns if 'Number of doctoral' in col]
    raw_env_doctoral['num_doc_degrees_total'] = raw_env_doctoral[number_cols].sum(axis=1)

    # Research income data
    raw_env_income = pd.read_excel(raw_env_path, sheet_name="ResearchIncome", skiprows=4)
    raw_env_income = format_ids(raw_env_income)
    raw_env_income = raw_env_income.rename(
        columns={'Average income for academic years 2013-14 to 2019-20': 'av_income',
                 'Total income for academic years 2013-14 to 2019-20': 'tot_income'})
    tot_inc = raw_env_income[raw_env_income['Income source'] == 'Total income']

    # Research income in-kind data
    raw_env_income_inkind = pd.read_excel(raw_env_path, sheet_name="ResearchIncomeInKind", skiprows=4)
    raw_env_income_inkind = format_ids(raw_env_income_inkind)
    raw_env_income_inkind = raw_env_income_inkind.rename(
        columns={'Total income for academic years 2013-14 to 2019-20': 'tot_inc_kind'})

    tot_inc_kind = raw_env_income_inkind.loc[raw_env_income_inkind['Income source'] == 'Total income-in-kind']

    ## Merge all dept level data together
    raw_dep = merge_ins_uoa(
        raw_results[['inst_id', 'Institution name', 'uoa_id', 'fte', 'fte_pc']].drop_duplicates(),
        wide_score_card)
    raw_dep = merge_ins_uoa(raw_dep,
                            raw_env_doctoral[['inst_id', 'uoa_id', 'num_doc_degrees_total']])
    raw_dep = merge_ins_uoa(raw_dep,
                            tot_inc[['inst_id', 'uoa_id', 'av_income', 'tot_income']])
    raw_dep = merge_ins_uoa(raw_dep,
                            tot_inc_kind[['inst_id', 'uoa_id', 'tot_inc_kind']])

    raw_dep['ICS_GPA'] = (pd.to_numeric(raw_dep['4*_Impact'], errors='coerce') * 4 +
                          pd.to_numeric(raw_dep['3*_Impact'], errors='coerce') * 3 +
                          pd.to_numeric(raw_dep['2*_Impact'], errors='coerce') * 2 +
                          pd.to_numeric(raw_dep['1*_Impact'], errors='coerce')) / 100
    raw_dep['Environment_GPA'] = (pd.to_numeric(raw_dep['4*_Environment'], errors='coerce') * 4 +
                                  pd.to_numeric(raw_dep['3*_Environment'], errors='coerce') * 3 +
                                  pd.to_numeric(raw_dep['2*_Environment'], errors='coerce') * 2 +
                                  pd.to_numeric(raw_dep['1*_Environment'], errors='coerce')) / 100
    raw_dep['Output_GPA'] = (pd.to_numeric(raw_dep['4*_Outputs'], errors='coerce') * 4 +
                             pd.to_numeric(raw_dep['3*_Outputs'], errors='coerce') * 3 +
                             pd.to_numeric(raw_dep['2*_Outputs'], errors='coerce') * 2 +
                             pd.to_numeric(raw_dep['1*_Outputs'], errors='coerce')) / 100
    raw_dep['Overall_GPA'] = (pd.to_numeric(raw_dep['4*_Overall'], errors='coerce') * 4 +
                              pd.to_numeric(raw_dep['3*_Overall'], errors='coerce') * 3 +
                              pd.to_numeric(raw_dep['2*_Overall'], errors='coerce') * 2 +
                              pd.to_numeric(raw_dep['1*_Overall'], errors='coerce')) / 100
    extracted = raw_dep['uoa_id'].str.extract(r'([-]?\d+)', expand=False)
    raw_dep['uoa_id'] = pd.to_numeric(extracted, errors='coerce').astype('Int64')
    df_merge = pd.merge(df_clean, raw_dep, how='left',
                        left_on=['Q3', 'Q8_uoa'],
                        right_on=['Institution name', 'uoa_id']
                        )
    logging.info(f"We have {len(df_merge[(df_merge['ICS_GPA'].isnull())])} null ICS_GPA")

    temp = df_merge[(df_merge['ICS_GPA'].isnull())]
    logging.info(f"OK, we have {len(temp)} rows with ICS_GPA to begin with")
    logging.info('Lets drop those without Q3 (institution)')
    temp = temp[temp['Q3'].notnull()]
    logging.info(f"We now have {len(temp)} rows with ICS_GPA")

    logging.info("Lets drop institution =='Other'")
    temp = temp[temp['Q3'] != 'Other']
    logging.info(f"We now have {len(temp)} rows with ICS_GPA")

    logging.info('Lets drop those without a UOA')
    temp = temp[temp['Q8_uoa'].notnull()]
    logging.info(f"We now have {len(temp)} rows with ICS_GPA")
    out_path = '../data/to_check/institutions_didnt_submit_to_uoa.csv'
    logging.info(f'Saving these out to {out_path}: Check these people were in universities which didnt to submit to relevant UOA.')
    temp.to_csv(out_path)
    classification = pd.read_csv('../data/manual_classification/for_merge/classification.csv', index_col=None)
    df_merge = pd.merge(df_merge, classification, how='left', left_on='PIPD', right_on='Q28')
    df_manual = pd.read_csv('../data/lookup/all_unis_manual.csv', index_col=None)
    df_merge = pd.merge(df_merge, df_manual, how='left', on='Q3')
    df_merge['is_oxbridge'] = np.where(df_merge['Q1'] == 'Other', 0, df_merge['is_oxbridge'])
    df_merge['is_redbrick'] = np.where(df_merge['Q1'] == 'Other', 0, df_merge['is_redbrick'])
    out_path = '../data/wrangled/df_clean_merged.csv'
    logging.info('Data getting saved to: ' + out_path)
    df_merge.to_csv(out_path)
    logging.info(f"Length of df_merge is {len(df_merge)}")


if __name__ == "__main__":
    build_dataset()
