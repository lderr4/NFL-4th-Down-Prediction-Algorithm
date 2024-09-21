import pandas as pd
from constants import keep_cols, drop_cols, dataset_csv_path
import nfl_data_py as nfl
import os


def partition(data, play_type="pass"):
    return data[data['play_type'] == play_type].drop('play_type', axis=1)

def import_dataset(years):
    
    raw_dataset = nfl.import_pbp_data(years=years)
    return raw_dataset

def align_spread(row): 
    if row["posteam_is_home"] is False:
        return row['spread_line'] * -1
    else:
        return row['spread_line']
    
def align_total_epa(row):
    if row['posteam_is_home'] is True:
        return row['total_home_epa']
    else:
        return row['total_away_epa']
    
def align_rush_epa(row):
    if row['posteam_is_home'] is True:
        return row['total_home_rush_epa']
    else:
        return row['total_away_rush_epa']
    
def align_pass_epa(row):
    if row['posteam_is_home'] is True:
        return row['total_home_pass_epa']
    else:
        return row['total_away_pass_epa']
    
def process_raw_dataset_to_csv(raw_dataset):

    raw_dataset['game_date'] = pd.to_datetime(raw_dataset['game_date'])
    raw_dataset['year'] = raw_dataset['game_date'].dt.year    
    df = raw_dataset[keep_cols]
    del raw_dataset
    df = df[df['down'] == 4]

    df['posteam_is_home'] = df.apply(lambda row: row['posteam'] == row['home_team'], axis=1)
    df['posteam_spread_line'] = df.apply(align_spread, axis=1)
    df['wpa_avg'] = (df['wpa'] + df['vegas_wpa']) / 2
    df['wp_avg'] = df.apply(lambda row: (row['wp'] + row["vegas_wp"]) / 2, axis=1)
    df['is_reg_season'] = df['season_type'].apply(lambda x: True if x == "REG" else False)
    df['wpa_difference'] = abs(df['wpa'] - df['vegas_wpa'])
    df['wp_difference'] = abs(df['wp'] - df['vegas_wp'])

    df['posteam_total_epa'] = df.apply(align_total_epa, axis=1)
    df['posteam_total_rush_epa'] = df.apply(align_rush_epa, axis=1)
    df['posteam_total_pass_epa'] = df.apply(align_total_epa, axis=1)


    df['posteam_epa_per_sec'] = df.apply(lambda row: row['posteam_total_epa'] / (3600 - row['game_seconds_remaining']), axis=1)
    df['posteam_rush_epa_per_sec'] = df.apply(lambda row: row['posteam_total_rush_epa'] / (3600 - row['game_seconds_remaining']), axis=1)
    df['posteam_pass_epa_per_sec'] = df.apply(lambda row: row['posteam_total_pass_epa'] / (3600 - row['game_seconds_remaining']), axis=1)

    df.drop(columns=drop_cols, inplace=True)
    df.to_csv(dataset_csv_path)
    return df

def get_dataset(years):
    
    if os.path.exists(dataset_csv_path):
        print("Loading data from csv.")
        dataset = pd.read_csv(dataset_csv_path)
    else:
        print("Dataset csv not found. Running Download script.")
        dataset = import_dataset(years)
        dataset = process_raw_dataset_to_csv(dataset)
    return dataset

def partition_dataset_by_play_type(dataset):

    passing = partition(dataset,'pass')
    run = partition(dataset,'run')
    fg = partition(dataset,'field_goal')
    punt = partition(dataset,'punt')

    partitioned_dataset = [("run", run) , 
                        ("pass", passing), 
                        ("fg", fg), 
                        ("punt", punt)]
    return partitioned_dataset

def prep_for_classifier(dataset):
    dataset = dataset.dropna(subset=['play_type'])
    dataset.drop(dataset[dataset['play_type'] == 'no_play'].index, inplace=True)
    dataset.drop(dataset[dataset['play_type'] == 'qb_kneel'].index, inplace=True)
    # dataset = dataset.drop('wpa_avg', axis=1)
    return dataset