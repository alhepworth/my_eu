#!/usr/bin/env python
# coding: utf-8

# # Build Data for the Map

# Check and merge datasets from:
#     cordis
#     creative
#     esif
#     fts
#     erasmus
#     nhs
#     nweurope
#     life

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from functools import reduce
import glob
import json
import os

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 50)


# ## Define Validity Checks

# In[ ]:


ukpostcodes = pd.read_csv('../postcodes/input/ukpostcodes.csv.gz')
ukpostcodes.shape


# In[ ]:


def validate_postcodes(df):
    assert 'postcode' in df.columns
    assert (~df['postcode'].isin(ukpostcodes.postcode)).sum() == 0
    
def validate_date_range(df):
    assert 'start_date' in df.columns
    assert 'end_date' in df.columns
    assert df['start_date'].dtype == 'datetime64[ns]'
    assert df['end_date'].dtype == 'datetime64[ns]'
    assert (df['start_date'] > df['end_date']).sum() == 0


# ## Load Cleaned Data

# ### CORDIS

# In[ ]:


fp7_organizations = pd.read_pickle('../cordis/output/fp7_organizations.pkl.gz')
validate_postcodes(fp7_organizations)
fp7_organizations.head()


# In[ ]:


fp7_projects = pd.read_pickle('../cordis/output/fp7_projects.pkl.gz')
validate_date_range(fp7_projects)
fp7_projects.head()


# In[ ]:


fp7 = pd.merge(
    fp7_projects, fp7_organizations,
    left_on='rcn', right_on='project_rcn', validate='1:m'
)
fp7['my_eu_id'] = 'fp7_' + fp7.project_rcn.astype('str') + '_' + fp7.organization_id.astype('str')
fp7.head()


# In[ ]:


h2020_organizations = pd.read_pickle('../cordis/output/h2020_organizations.pkl.gz')
validate_postcodes(h2020_organizations)
h2020_organizations.head()


# In[ ]:


h2020_projects = pd.read_pickle('../cordis/output/h2020_projects.pkl.gz')
validate_date_range(h2020_projects)
h2020_projects.head()


# In[ ]:


h2020 = pd.merge(
    h2020_projects, h2020_organizations,
    left_on='rcn', right_on='project_rcn', validate='1:m'
)
h2020['my_eu_id'] = 'h2020_' + h2020.project_rcn.astype('str') + '_' + h2020.organization_id.astype('str')

# no briefs available for H2020
h2020['related_report_title'] = float('nan')
h2020['brief_title'] = float('nan')
h2020['teaser'] = float('nan')
h2020['article'] = float('nan')
h2020['image_path'] = float('nan')

h2020.head()


# In[ ]:


assert set(fp7.columns) == set(h2020.columns)


# In[ ]:


cordis = pd.concat([fp7, h2020[fp7.columns]])
cordis.shape


# In[ ]:


cordis['total_cost_gbp'] = (cordis.total_cost_eur * cordis.eur_gbp).round()
cordis['max_contribution_gbp'] = (cordis.max_contribution_eur * cordis.eur_gbp).round()
cordis['contribution_gbp'] = (cordis.contribution_eur * cordis.eur_gbp).round()
cordis.head()


# In[ ]:


cordis.describe()


# In[ ]:


(cordis.contribution_eur > cordis.total_cost_eur + 0.1).sum()


# In[ ]:


[
    cordis.start_date.isna().sum(),
    cordis.end_date.isna().sum()
]


# ### Creative Europe

# In[ ]:


creative_organisations = pd.read_pickle('../creative/output/creative_europe_organisations.pkl.gz')
creative_organisations.shape


# In[ ]:


creative_projects = pd.read_pickle('../creative/output/creative_europe_projects.pkl.gz')
creative_projects.shape


# In[ ]:


creative = pd.merge(creative_projects, creative_organisations, on='project_number', validate='1:m')
creative.shape


# In[ ]:


validate_postcodes(creative)
validate_date_range(creative)
creative['max_contribution_gbp'] = (creative.max_contribution_eur * creative.eur_gbp).round()
creative['my_eu_id'] =     'creative_' + creative.project_number + '_' +     creative.partner_number.apply('{:.0f}'.format).    str.replace('nan', 'coordinator', regex=False)
assert creative.shape[0] == creative.my_eu_id.unique().shape[0]
creative.head()


# In[ ]:


creative.results_available.value_counts()


# In[ ]:


creative.results_url[0]


# In[ ]:


[creative.start_date.isna().sum(), creative.end_date.isna().sum()]


# ### ESIF (ESF/ERDF)

# In[ ]:


esif_england = pd.read_pickle('../esif/output/esif_england_2014_2020.pkl.gz')
validate_postcodes(esif_england)
validate_date_range(esif_england)
esif_england.head()


# In[ ]:


esif_ni = pd.read_pickle('../esif/output/esif_ni_2014_2020.pkl.gz')
validate_postcodes(esif_ni)
validate_date_range(esif_ni)
esif_ni.head()


# In[ ]:


esif_scotland = pd.read_pickle('../esif/output/esif_scotland.pkl.gz')
validate_postcodes(esif_scotland)
validate_date_range(esif_scotland)
esif_scotland.head()


# In[ ]:


esif_wales = pd.read_pickle('../esif/output/esif_wales.pkl.gz')
validate_postcodes(esif_wales)
validate_date_range(esif_wales)
esif_wales.head()


# In[ ]:


assert set(esif_england.columns) == set(esif_ni.columns)
assert set(esif_england.columns) == set(esif_scotland.columns)
assert set(esif_england.columns) == set(esif_wales.columns)
esif_columns = esif_england.columns
esif = pd.concat([
    esif_england,
    esif_ni[esif_columns],
    esif_scotland[esif_columns],
    esif_wales[esif_columns]
])
esif.shape


# In[ ]:


[esif.start_date.isna().sum(), esif.end_date.isna().sum()]


# ### FTS

# In[ ]:


fts_2016 = pd.read_pickle('../fts/output/fts_2016.pkl.gz')
validate_postcodes(fts_2016)
fts_2016['year'] = 2016
fts_2016.head()


# In[ ]:


fts_2017 = pd.read_pickle('../fts/output/fts_2017.pkl.gz')
validate_postcodes(fts_2017)
fts_2017['year'] = 2017
fts_2017.head()


# In[ ]:


fts = pd.concat([fts_2016, fts_2017])
fts['amount_gbp'] = (fts.amount * fts.eur_gbp).round()
fts['total_amount_gbp'] = (fts.total_amount_eur * fts.eur_gbp).round()
fts.shape


# ### Erasmus

# In[ ]:


erasmus_organisations = pd.read_pickle('../erasmus/output/erasmus_mobility_organisations.pkl.gz')
erasmus_organisations.shape


# In[ ]:


erasmus_projects = pd.read_pickle('../erasmus/output/erasmus_mobility_projects.pkl.gz')
erasmus_projects.shape


# In[ ]:


erasmus = pd.merge(erasmus_projects, erasmus_organisations, on='project_identifier', validate='1:m')
erasmus.shape


# In[ ]:


validate_postcodes(erasmus)

erasmus['max_contribution_gbp'] = (erasmus.max_contribution_eur * erasmus.eur_gbp).round()
erasmus['my_eu_id'] =     'erasmus_' + erasmus.project_identifier + '_' +     erasmus.partner_number.apply('{:.0f}'.format).    str.replace('nan', 'coordinator', regex=False)
assert erasmus.shape[0] == erasmus.my_eu_id.unique().shape[0]
erasmus.head()


# ### NHS

# In[ ]:


nhs_staff = pd.read_pickle('../nhs/output/staff.pkl.gz')
nhs_hospital_postcodes = pd.read_pickle('../nhs/output/hospital_postcodes.pkl.gz')
validate_postcodes(nhs_hospital_postcodes)
[nhs_staff.shape, nhs_hospital_postcodes.shape, nhs_hospital_postcodes.hospital_organisation.nunique()]


# In[ ]:


# dummy amount so we can put it in
nhs_hospital_postcodes['zero'] = 0


# In[ ]:


nhs_hospital_postcodes['my_eu_id'] =     'nhs_' + nhs_hospital_postcodes.hospital_organisation
nhs_hospital_postcodes.head()


# ### Interreg NW Europe

# In[ ]:


nweurope = pd.read_pickle('../nweurope/output/interreg_beneficiaries.pkl.gz')
validate_postcodes(nweurope)
nweurope.head(2)


# ### LIFE

# In[ ]:


life = pd.read_pickle('../life/output/life.pkl.gz')
validate_postcodes(life)


# In[ ]:


life['total_budget_gbp'] = (life.total_budget_eur * life.eur_gbp).round()
life['eu_contribution_gbp'] = (life.eu_contribution_eur * life.eur_gbp).round()
life.shape


# ## Idea 1: All Points on Map, Data by District
# 
# This should make the map look fairly similar to how it looks now, so it seems like a good starting point.

# In[ ]:


ALL_PLACES = [
    (cordis, 'contribution_gbp', 'money'),
    (creative, 'max_contribution_gbp', 'money'),
    (esif, 'eu_investment', 'money'),
    (fts.drop('amount', axis=1), 'amount_gbp', 'money'),
    (erasmus, 'max_contribution_gbp', 'money'),
    (nhs_hospital_postcodes, 'zero', 'hospital'),
    (nweurope, 'funding', 'money'),
    (life, 'eu_contribution_gbp', 'money')
]


# GeoJSON is very inefficient for representing a bunch of points, so let's use a relatively simple packed format.
# ```
# min_longitude min_latitude
# outcode incode delta_longitude delta_latitude incode delta_longitude delta_latitude
# ```
# We need [about 4 decimal places](https://gis.stackexchange.com/questions/8650/measuring-accuracy-of-latitude-and-longitude).

# In[ ]:


def add_outward_and_inward_codes(df):
    df['outward_code'] = df.postcode.str.split(' ').str[0]
    df['inward_code'] = df.postcode.str.split(' ').str[1]
    return df

def add_packed_icon_mask(df):
    def pack_icon_mask(icons):
        # So far we just have one bit in our mask; we may have more.
        if 'hospital' in icons:
            return 1
        else:
            return 0
    MASK_BITS = 1
    assert df.amount.max() < 2**(32 - MASK_BITS)
    df['icon_mask'] = df.icons.apply(pack_icon_mask).astype('uint32')
    df['packed_amount'] = np.bitwise_or(
        np.left_shift(df.amount, MASK_BITS), df.icon_mask
    )
    return df

def pack_geocoded_postcodes(dfs):
    all_postcode_amounts = pd.concat([
        df.rename(columns={amount_column: 'amount'}).assign(icons=icon)\
        [['postcode', 'amount', 'icons']]
        for df, amount_column, icon in dfs
    ])
    postcode_amounts = all_postcode_amounts.groupby('postcode').        aggregate({'amount': sum, 'icons': lambda icons: set(icons)})
    postcode_amounts.reset_index(inplace=True)
    postcode_amounts.amount = postcode_amounts.amount.astype('uint32')
    add_outward_and_inward_codes(postcode_amounts)
    add_packed_icon_mask(postcode_amounts)
    
    geocoded_postcodes = pd.merge(postcode_amounts, ukpostcodes, validate='1:1')
    
    min_longitude = geocoded_postcodes.longitude.min()
    min_latitude = geocoded_postcodes.latitude.min()
    
    geocoded_postcodes['delta_longitude'] = geocoded_postcodes.longitude - min_longitude
    geocoded_postcodes['delta_latitude'] = geocoded_postcodes.latitude - min_latitude
    
    return {
        'min_longitude': min_longitude,
        'min_latitude': min_latitude,
        'geocoded_postcodes': geocoded_postcodes
    }

packed_postcodes = pack_geocoded_postcodes(ALL_PLACES)
[
    packed_postcodes['min_longitude'],
    packed_postcodes['min_latitude'],
    packed_postcodes['geocoded_postcodes'].shape[0]
]


# In[ ]:


packed_postcodes['geocoded_postcodes'].head()


# In[ ]:


def make_packed_postcode_json(packed_postcodes):
    packed_postcodes = packed_postcodes.copy()
   
    grouped_postcodes = packed_postcodes['geocoded_postcodes'].        sort_values('outward_code').groupby('outward_code')
     
    def make_code_tuples(row):
        coordinate = '{0:.4f}'
        return [
            row['inward_code'],
            float(coordinate.format(row['delta_longitude'])),
            float(coordinate.format(row['delta_latitude'])),
            row['packed_amount']
        ]
    
    postcodes = {}
    for outward_code, group in grouped_postcodes:
        postcodes[outward_code] = [
            x
            for code in group.sort_values('inward_code').apply(make_code_tuples, axis=1)
            for x in code
        ]

    min_coordinate = '{0:.6f}'
    return {
        'min_longitude': float(min_coordinate.format(packed_postcodes['min_longitude'])),
        'min_latitude': float(min_coordinate.format(packed_postcodes['min_latitude'])),
        'postcodes': postcodes
    }

with open('output/packed_postcodes.data.json', 'w') as file:
    json.dump(make_packed_postcode_json(packed_postcodes), file, sort_keys=True)


# ### Some Aggregates

# In[ ]:


[place[0].shape[0] for place in ALL_PLACES]


# In[ ]:


sum(place[0].shape[0] for place in ALL_PLACES)


# ### Data by District
# 
# #### CORDIS

# In[ ]:


MAX_PROJECTS = 50

# Dump to JSON using pandas, because it puts in nulls instead of NaNs for
# missing values. Then load the JSON into dicts for 
def make_district_data_json(df, name):
    key = ['outwardCode', 'inwardCode']
    def to_json(group):
        group.drop(key, axis=1, inplace=True)
        return json.loads(group.to_json(orient='split', index=False))
    grouped = df.groupby(key).apply(to_json)
    
    grouped = grouped.reset_index()
    grouped.columns = key + [name]
    
    for _key, row in grouped.iterrows():
        count = len(row[name]['data'])
        if count > MAX_PROJECTS:
            row[name]['data'] = row[name]['data'][0:MAX_PROJECTS]
            row[name]['extra'] = count - MAX_PROJECTS
        
    return grouped

def make_cordis_district_data(cordis):
    cordis = add_outward_and_inward_codes(cordis.copy())

    cordis = cordis[[
        'outward_code',
        'inward_code',
        'start_date', 'end_date',
        'title',
        'name', # of organization
        'objective',
        'contribution_gbp',
        'total_cost_gbp',
        'num_countries',
        'num_organizations',
        'acronym',
        'project_url',
        'organization_url',
        'image_path',
        'my_eu_id'
    ]]
    
    cordis.rename({
        'outward_code': 'outwardCode',
        'inward_code': 'inwardCode',
        'start_date': 'startDate',
        'end_date': 'endDate',
        'title': 'projectTitle',
        'name': 'organisationName',
        'contribution_gbp': 'contribution',
        'total_cost_gbp': 'totalCost',
        'num_countries': 'numCountries',
        'num_organizations': 'numOrganisations',
        'project_url': 'projectUrl',
        'organization_url': 'organisationUrl',
        'image_path': 'imagePath',
        'my_eu_id': 'myEuId'
    }, axis=1, inplace=True)
    
    cordis.sort_values(
        ['outwardCode', 'inwardCode', 'contribution'],
        ascending=[True, True, False],
        inplace=True
    )
    
    return make_district_data_json(cordis, 'cordis')

cordis_district_data = make_cordis_district_data(cordis)
cordis_district_data.head()


# #### Creative Europe

# In[ ]:


def make_creative_district_data(creative):
    creative = add_outward_and_inward_codes(creative.copy())
    
    coordinators = creative[creative.organisation_coordinator]
    coordinators = coordinators[['project_number', 'organisation_name']]
    creative = pd.merge(
        creative, coordinators,
        how='left', on='project_number', suffixes=('', '_coordinator'))

    creative = creative[[
        'outward_code',
        'inward_code',
        'start_date', 'end_date',
        'project',
        'organisation_name',
        'max_contribution_gbp',
        'num_countries',
        'num_organisations',
        'summary',
        'organisation_website',
        'organisation_name_coordinator',
        'my_eu_id'
    ]]
    
    creative.rename({
        'outward_code': 'outwardCode',
        'inward_code': 'inwardCode',
        'start_date': 'startDate',
        'end_date': 'endDate',
        'organisation_name': 'organisationName',
        'max_contribution_gbp': 'maxContribution',
        'num_countries': 'numCountries',
        'num_organisations': 'numOrganisations',
        'organisation_website': 'organisationWebsite',
        'organisation_name_coordinator': 'coordinatorName',
        'my_eu_id': 'myEuId'
    }, axis=1, inplace=True)
    
    creative.sort_values(
        ['outwardCode', 'inwardCode', 'maxContribution'],
        ascending=[True, True, False],
        inplace=True
    )
    
    return make_district_data_json(creative, 'creative')

creative_district_data = make_creative_district_data(creative)
creative_district_data.head()


# #### ESIF

# In[ ]:


def make_esif_district_data(esif):
    esif = add_outward_and_inward_codes(esif.copy())
    esif = esif[[
        'outward_code',
        'inward_code',
        'start_date', 'end_date',
        'project',
        'beneficiary',
        'summary',
        'funds',
        'eu_investment',
        'project_cost',
        'my_eu_id'
    ]]
    
    esif.rename({
        'outward_code': 'outwardCode',
        'inward_code': 'inwardCode',
        'start_date': 'startDate',
        'end_date': 'endDate',
        'project': 'projectTitle',
        'beneficiary': 'organisationName',
        'eu_investment': 'euInvestment',
        'project_cost': 'projectCost',
        'my_eu_id': 'myEuId'
    }, axis=1, inplace=True)
    
    esif.sort_values(
        ['outwardCode', 'inwardCode', 'euInvestment'],
        ascending=[True, True, False],
        inplace=True
    )
    
    return make_district_data_json(esif, 'esif')

esif_district_data = make_esif_district_data(esif)
esif_district_data.head()


# #### FTS

# In[ ]:


fts.columns


# In[ ]:


def make_fts_district_data(fts):
    fts = add_outward_and_inward_codes(fts.copy())
    fts = fts[[
        'outward_code',
        'inward_code',
        'year',
        'beneficiary',
        'amount_gbp',
        'budget_line_name_and_number',
        'my_eu_id'
    ]]
    
    fts.rename({
        'outward_code': 'outwardCode',
        'inward_code': 'inwardCode',
        'amount_gbp': 'amount',
        'budget_line_name_and_number': 'budgetLineNameAndNumber',
        'my_eu_id': 'myEuId'
    }, axis=1, inplace=True)
    
    fts.sort_values(
        ['outwardCode', 'inwardCode', 'amount'],
        ascending=[True, True, False],
        inplace=True
    )
    
    return make_district_data_json(fts, 'fts')

fts_district_data = make_fts_district_data(fts)
fts_district_data.head()


# #### Erasmus

# In[ ]:


def make_erasmus_district_data(erasmus):
    erasmus = add_outward_and_inward_codes(erasmus.copy())

    coordinators = erasmus[erasmus.organisation_coordinator]
    coordinators = coordinators[['project_identifier', 'organisation_name']]
    erasmus = pd.merge(
        erasmus, coordinators,
        how='left', on='project_identifier', suffixes=('', '_coordinator'))

    erasmus = erasmus[[
        'outward_code',
        'inward_code',
        'call_year',
        'project',
        'organisation_name',
        'max_contribution_gbp',
        'num_countries',
        'num_organisations',
        'summary',
        'organisation_website',
        'organisation_name_coordinator',
        'my_eu_id'
    ]]
    
    erasmus.rename({
        'outward_code': 'outwardCode',
        'inward_code': 'inwardCode',
        'call_year': 'callYear',
        'organisation_name': 'organisationName',
        'max_contribution_gbp': 'maxContribution',
        'num_countries': 'numCountries',
        'num_organisations': 'numOrganisations',
        'organisation_website': 'organisationWebsite',
        'organisation_name_coordinator': 'coordinatorName',
        'my_eu_id': 'myEuId'
    }, axis=1, inplace=True)
    
    erasmus.sort_values(
        ['outwardCode', 'inwardCode', 'maxContribution'],
        ascending=[True, True, False],
        inplace=True
    )
    
    return make_district_data_json(erasmus, 'erasmus')

erasmus_district_data = make_erasmus_district_data(erasmus)
erasmus_district_data.head()


# #### NHS
# 
# Just store the organisation key for now. The data doesn't really fit with the district model.

# In[ ]:


def make_nhs_district_data(nhs_hospital_postcodes, nhs_staff):
    nhs = add_outward_and_inward_codes(pd.merge(
        nhs_hospital_postcodes, nhs_staff, on='organisation', validate='m:1'
    ))
    
    nhs.sort_values(
        ['outward_code', 'inward_code', 'eu_nurses'],
        ascending=[True, True, False],
        inplace=True
    )

    nhs = nhs[[
        'outward_code',
        'inward_code',
        'organisation',
        'hospital_name',
        'my_eu_id'
    ]]
    
    nhs.rename({
        'outward_code': 'outwardCode',
        'inward_code': 'inwardCode',
        'hospital_name': 'hospitalName',
        'my_eu_id': 'myEuId'
    }, axis=1, inplace=True)
    
    return make_district_data_json(nhs, 'nhs')

nhs_district_data = make_nhs_district_data(nhs_hospital_postcodes, nhs_staff)
nhs_district_data.head()


# #### Interreg NW Europe

# In[ ]:


nweurope.columns


# In[ ]:


nweurope.head(2)


# In[ ]:


def make_nweurope_district_data(nweurope):
    nweurope = add_outward_and_inward_codes(nweurope.copy())
    nweurope = nweurope[[
        'outward_code',
        'inward_code',
        'beneficiary',
        'project',
        'start_date',
        'end_date',
        'funding',
        'union_cofinancing',
        'nwreg_benefs_id'
    ]]
    
    nweurope.rename({
        'outward_code': 'outwardCode',
        'inward_code': 'inwardCode',
        'start_date': 'startDate',
        'end_date': 'endDate',
        'union_cofinancing': 'unionCofinancing',
        'nwreg_benefs_id': 'myEuId'
    }, axis=1, inplace=True)
    
    nweurope.sort_values(
        ['outwardCode', 'inwardCode', 'unionCofinancing'],
        ascending=[True, True, False],
        inplace=True
    )
    
    return make_district_data_json(nweurope, 'nweurope')

nweurope_district_data = make_nweurope_district_data(nweurope)
nweurope_district_data.head()


# #### LIFE

# In[ ]:


life.columns


# In[ ]:


def make_life_district_data(life):
    life = add_outward_and_inward_codes(life.copy())
    life = life[[
        'outward_code',
        'inward_code',
        'year',
        'project_title',
        'coordinator',
        'total_budget_gbp',
        'eu_contribution_gbp',
        'background',
        'project_url',
        'website',
        'my_eu_id'
    ]]
    
    life.rename({
        'outward_code': 'outwardCode',
        'inward_code': 'inwardCode',
        'total_budget_gbp': 'amount',
        'eu_contribution_gbp': 'euContribution',
        'my_eu_id': 'myEuId'
    }, axis=1, inplace=True)
    
    life.sort_values(
        ['outwardCode', 'inwardCode', 'amount'],
        ascending=[True, True, False],
        inplace=True
    )
    
    return make_district_data_json(life, 'life')

life_district_data = make_life_district_data(life)
life_district_data.head()


# ### Save Data
# 
# ```
# districtData.postcodes['A11'].cordis.data
# districtData.postcodes['A11'].cordis.extra
# ```

# In[ ]:


ALL_DISTRICT_DATA = reduce(
    lambda left, right: pd.merge(left, right, on=('outwardCode', 'inwardCode'), how='outer'), [
    cordis_district_data,
    creative_district_data,
    esif_district_data,
    fts_district_data,
    erasmus_district_data,
    nhs_district_data,
    nweurope_district_data,
    life_district_data    
])
ALL_DATASET_NAMES = list(set(ALL_DISTRICT_DATA.columns) - set(['outwardCode', 'inwardCode']))
    
def save_columns():
    def find_columns(data):
        return data[~data.isna()].iloc[0]['columns']
  
    columns = {
        dataset_name: find_columns(ALL_DISTRICT_DATA[dataset_name])
        for dataset_name in ALL_DATASET_NAMES
    }
    
    with open(os.path.join('output', 'district_columns.json'), 'w') as file:
        json.dump(columns, file, sort_keys=True)
save_columns()


# In[ ]:


ALL_DISTRICT_DATA[(~ALL_DISTRICT_DATA[ALL_DATASET_NAMES].isna()).sum(axis=1) > 4][['outwardCode', 'inwardCode']].head()


# In[ ]:


def merge_district_data(all_data):
    def find_dataset_data(row):
        dataset_data = row.drop(['outwardCode', 'inwardCode']).dropna().to_dict()
        return {
            dataset: { k: v for k, v in json.items() if k != 'columns' }
            for dataset, json in dataset_data.items()
        }
    
    def make_merged_district_data(outward_code, group):
        return {
            'outwardCode': outward_code,
            'postcodes': {
                row.inwardCode: find_dataset_data(row)
                for _index, row in group.iterrows()
            }
        }
    
    return {
        outward_code: make_merged_district_data(outward_code, group)
        for outward_code, group in all_data.groupby('outwardCode')
    }

district_data = merge_district_data(ALL_DISTRICT_DATA)
district_data['CA4']


# In[ ]:


OUTPUT_DISTRICT_PATH = 'output/district'

def list_district_data(path):
    return glob.glob(os.path.join(path, '*.data.json'))

def clear_district_data(path):
    for f in list_district_data(path):
        os.remove(f)

def write_district_data(district_data, path):
    os.makedirs(path, exist_ok=True)
    clear_district_data(path)
    for outward_code, data in district_data.items():
        output_pathname = os.path.join(path, outward_code + '.data.json')
        with open(output_pathname, 'w') as file:
            json.dump(data, file, sort_keys=True)
write_district_data(district_data, OUTPUT_DISTRICT_PATH)


# In[ ]:


def find_district_data_stats():
    files = list_district_data(OUTPUT_DISTRICT_PATH)
    return pd.DataFrame({
        'file': [file for file in files],
        'byte_size': [os.stat(file).st_size for file in files]
    })
district_data_stats = find_district_data_stats()
district_data_stats.describe()


# In[ ]:


district_data_stats.byte_size.sum() / 1024 / 1024


# In[ ]:


district_data_stats[district_data_stats.byte_size > 1024*1024]


# In[ ]:


district_data_stats.describe().hist()


# #### Data Index
# 
# Generate a JS file that webpack can use to make paths for all of the data files.

# In[ ]:


def write_district_data_js():
    data_files = list_district_data(OUTPUT_DISTRICT_PATH)
    
    def make_require(data_file):
        basename = os.path.basename(data_file)
        pathname = os.path.join('.', 'district', basename)
        outward_code = basename.split('.')[0]
        return "  {}: require('{}')".format(outward_code, pathname)

    with open('output/district.js', 'w') as file:
        file.write('// NB: This file is generated automatically. Do not edit.\n')
        file.write('export default {\n')
        requires = [
            make_require(data_file)
            for data_file in sorted(data_files)
        ]
        file.write(',\n'.join(requires))
        file.write('\n}\n')
write_district_data_js()


# #### NHS Data

# In[ ]:


[nhs_staff.shape[0], nhs_hospital_postcodes.shape[0]]


# In[ ]:


nhs_hospital_postcodes.groupby('hospital_organisation').organisation.count().max()


# In[ ]:


def write_nhs_staff_data():
    with open('output/nhs_staff.json', 'w') as file:
        file.write(
            nhs_staff.sort_values('organisation', ascending=True).\
            to_json(orient='split', index=False))
write_nhs_staff_data()


# In[ ]:




