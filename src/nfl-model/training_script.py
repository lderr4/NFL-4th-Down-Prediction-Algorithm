# from robo_coach import robo_coach

# rb = robo_coach()

# rb.train_models()

from preprocessing import get_dataset
from robo_coach import robo_coach


rb = robo_coach()
rb.load()

print("fg", rb.fg_columns)
print("pass", rb.pass_columns)
print("punt", rb.punt_columns)
print("run", rb.run_columns)
print("classifier", rb.classifier_columns)

'''
fg Index(['yardline_100', 'half_seconds_remaining', 'game_seconds_remaining',
       'drive', 'ydstogo', 'defteam_timeouts_remaining', 'score_differential',
       'no_score_prob', 'fg_prob', 'td_prob', 'ep', 'spread_line',
       'posteam_is_home', 'posteam_spread_line', 'wp_avg', 'wp_difference',
       'posteam_rush_epa_per_sec', 'posteam_pass_epa_per_sec'],
      dtype='object')
pass Index(['yardline_100', 'game_seconds_remaining', 'drive', 'ydsnet',
       'score_differential', 'fg_prob', 'drive_play_count', 'posteam_is_home',
       'wp_avg', 'wp_difference', 'posteam_pass_epa_per_sec'],
      dtype='object')
punt Index(['yardline_100', 'half_seconds_remaining', 'game_seconds_remaining',
       'drive', 'ydstogo', 'defteam_timeouts_remaining', 'score_differential',
       'no_score_prob', 'fg_prob', 'posteam_is_home', 'posteam_spread_line',
       'wp_avg', 'wp_difference', 'posteam_pass_epa_per_sec'],
      dtype='object')
run Index(['yardline_100', 'half_seconds_remaining', 'game_seconds_remaining',
       'drive', 'ydsnet', 'posteam_timeouts_remaining',
       'defteam_timeouts_remaining', 'score_differential', 'fg_prob', 'ep',
       'drive_play_count', 'spread_line', 'posteam_is_home',
       'posteam_spread_line', 'wp_avg', 'wp_difference',
       'posteam_rush_epa_per_sec', 'posteam_pass_epa_per_sec'],
      dtype='object')
'''