

SLIDER_DEFAULTS = {
        'week':( 1, 22, 1),
        'yardline_100':( 1, 99, 50),
        'half_seconds_remaining':( 0, 1800, 900),
        'game_seconds_remaining':( 0, 3600, 1800),
        'drive':( 1, 34, 1),
        'ydstogo':( 1, 43, 10),
        'ydsnet':( -99, 99, 0),
        'no_huddle':( 0, 1, 0),
        'posteam_timeouts_remaining':( 0, 3, 3),
        'defteam_timeouts_remaining':( 0, 3, 3),
        'score_differential':( -100, 100, 0),
        'no_score_prob':( 0.0, 1.0, 0.5),
        'fg_prob':( 0.0, 1.0, 0.5),
        'td_prob':( 0.0, 1.0, 0.5),
        'ep':( -6.0, 6.0, 0.0),
        'drive_play_count':( 0, 40, 20),
        'spread_line':( -50.0, 50.0, 0.0),
        'posteam_spread_line':( -50.0, 50.0, 0.0),
        'posteam_is_home':(0,1,0),
        'wpa_avg':( -1.0, 1.0, 0.0),
        'wp_avg':( 0.0, 1.0, 0.5),
        'wp_difference':( 0.0, 1.0, 0.5),
        'posteam_rush_epa_per_sec':( -1.0, 1.0, 0.0),
        'posteam_pass_epa_per_sec':( -1.0, 1.0, 0.0),
}


COLUMN_NAME_CLEANER = {
        'week': 'Week',
        'yardline_100': 'Distance to Goal',
        'half_seconds_remaining': 'Half Seconds Remaining',
        'game_seconds_remaining': 'Game Seconds Remaining',
        'drive': 'Drive Number',
        'ydstogo': 'Yards to Go',
        'ydsnet': 'Yards in Drive',
        'no_huddle': 'No Huddle',
        'posteam_timeouts_remaining': "Timeouts Remaining (Offense)",
        'defteam_timeouts_remaining': "Timeouts Remaining (Defense)",
        'score_differential': "Score Differential",
        'no_score_prob': "No Score for Rest of Half Probability",
        'fg_prob': 'Field Goal Probability',
        'td_prob': "Touchdown Probability",
        'ep': "Expected Points",
        'drive_play_count': "Drive Play Count",
        'spread_line': "Spread",
        'posteam_spread_line': "Spread Line",
        'wpa_avg': "Win Probability Added",
        'wp_avg': "Win Probability",
        'wp_difference': "Win Probability Uncertainty",
        'posteam_rush_epa_per_sec': "Expected Points Added on Rushing Plays per Second",
        'posteam_pass_epa_per_sec': "Expected Points Added on Passing Plays per Second",
        'posteam_is_home': 'Home Team',
        
        "field_goal": "Field Goal",
        "pass": "Pass",
        "punt": "Punt",
        "run": "Run"
}

BAR_CHART_DEFAULTS = {
    "field_goal":0,
    "pass":0,
    "punt":0,
    "run":0
}


API_URL = "http://nfl-model:8000"