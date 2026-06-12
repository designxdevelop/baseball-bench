# baseball_bench schema

## teams
- `team_id`: short identifier
- `city`
- `nickname`
- `league`
- `venue_name`

## players
- `player_id`
- `first_name`
- `last_name`
- `bats`
- `throws`
- `primary_position`

## batting
- `season`
- `team_id`
- `player_id`
- `games`
- `plate_appearances`
- `at_bats`
- `hits`
- `doubles`
- `triples`
- `home_runs`
- `walks`
- `strikeouts`
- `hit_by_pitch`
- `sacrifice_flies`

Derived metrics often used in queries:
- `singles = hits - doubles - triples - home_runs`
- `batting_average = hits / at_bats`
- `on_base_percentage = (hits + walks + hit_by_pitch) / (at_bats + walks + hit_by_pitch + sacrifice_flies)`
- `slugging = (singles + doubles * 2 + triples * 3 + home_runs * 4) / at_bats`
- `ops = on_base_percentage + slugging`

## pitching
- `season`
- `team_id`
- `player_id`
- `games`
- `games_started`
- `innings_pitched`
- `hits_allowed`
- `earned_runs`
- `home_runs_allowed`
- `walks`
- `strikeouts`

Derived metrics often used in queries:
- `era = earned_runs * 9 / innings_pitched`
- `k_per_9 = strikeouts * 9 / innings_pitched`
- `bb_per_9 = walks * 9 / innings_pitched`

## games
- `game_id`
- `season`
- `game_date`
- `home_team_id`
- `away_team_id`
- `home_score`
- `away_score`

## win_probabilities
- `inning`
- `half`
- `outs`
- `runner_state`: three-character base occupancy string, e.g. `100`, `011`
- `score_diff`: home score minus away score
- `home_win_prob`
