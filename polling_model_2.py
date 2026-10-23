import pandas as pd
import numpy as np
from datetime import datetime

TODAY = datetime(2026, 8, 26)
ELECTION_DAY = datetime(2026, 11, 3)
days_to_election = (ELECTION_DAY - TODAY).days
fundamental_uncertainty = 3.0 * np.sqrt(days_to_election / 365)

polls = pd.read_csv('combined_raw_polls.csv')
ratings = pd.read_csv('pollster-ratings-combined.csv')

# Parse date and calculate days ago
polls['Last Date'] = pd.to_datetime(polls['Last Date'], format='%m/%d/%y')
polls['days_ago'] = (TODAY - polls['Last Date']).dt.days

# Merge pollster ratings onto polls
polls = polls.merge(
    ratings[['pollster', 'POLLSCORE', 'bias_ppm']],
    left_on='Pollster Key',
    right_on='pollster',
    how='left'
)

polls = polls.dropna(subset=['State'])

# Convert bias_ppm to percentage points
polls['bias_pct'] = polls['bias_ppm'] / 100

# Adjust D and R for house effect
polls['D_adjusted'] = polls['D'] - polls['bias_pct']
polls['R_adjusted'] = polls['R'] + polls['bias_pct']

# Check for any polls missing a rating
missing = polls[polls['POLLSCORE'].isna()]
#if len(missing) > 0:
   # print(f"Warning: {len(missing)} polls missing a rating:")
   # print(missing[['State', 'Pollster', 'Pollster Key']])
#else:
   # print("All polls matched successfully")

HALF_LIFE_DAYS = 45

def recency_weight(days_ago):
    return np.exp(-days_ago / HALF_LIFE_DAYS)

def sample_size_weight(sample_size):
    return np.sqrt(sample_size) / np.sqrt(1000)

def quality_weight(pollscore):
    MIN_POLLSCORE = -1.5
    MAX_POLLSCORE = 1.0
    clamped = np.clip(pollscore, MIN_POLLSCORE, MAX_POLLSCORE)
    return (MAX_POLLSCORE - clamped) / (MAX_POLLSCORE - MIN_POLLSCORE)

# Apply weights
polls['w_recency'] = polls['days_ago'].apply(recency_weight)
polls['w_quality'] = polls['POLLSCORE'].apply(quality_weight)
polls['w_sample'] = polls['n'].apply(sample_size_weight)
polls['weight'] = polls['w_recency'] * polls['w_sample'] * polls['w_quality']

#print(polls[['State', 'Pollster', 'days_ago', 'n', 'w_recency', 'w_sample', 'w_quality', 'weight']])

def weighted_avg(group):
    group['weight_norm'] = group['weight'] / group['weight'].sum()
    dem = (group['D_adjusted'] * group['weight_norm']).sum()
    rep = (group['R_adjusted'] * group['weight_norm']).sum()
    margin = dem - rep
    
    # Weighted average MOE
    avg_moe = (group['MOE'] * group['weight_norm']).sum()
    
    # Effective sample size based on weights
    n_eff = 1 / (group['weight_norm'] ** 2).sum()
    
    # CI uses weighted MOE scaled by number of polls
    ci = avg_moe / np.sqrt(len(group))
    
    return pd.Series({
        'dem_estimate': round(dem, 2),
        'rep_estimate': round(rep, 2),
        'margin': round(margin, 2),
        'ci_95': round(ci * 1.96, 2),
        'margin_low': round(margin - ci * 1.96, 2),
        'margin_high': round(margin + ci * 1.96, 2),
        'n_polls': len(group)
    })

results = polls.groupby('State').apply(weighted_avg).reset_index()


#print(polls[['State', 'Pollster', 'weight']])

#print(polls[['State', 'Pollster', 'bias_ppm', 'bias_pct', 'D', 'D_adjusted', 'R', 'R_adjusted']])


from scipy import stats

def win_probability(margin, ci_95):
    polling_std = ci_95 / 1.96
    # Add fundamental uncertainty in quadrature (combines two independent error sources)
    total_std = np.sqrt(polling_std**2 + fundamental_uncertainty**2)
    prob_dem = stats.norm.cdf(margin / total_std)
    return round(prob_dem * 100, 1), total_std

results[['dem_win_prob', 'total_std']] = results.apply(
    lambda row: pd.Series(win_probability(row['margin'], row['ci_95'])), axis=1
)
results['rep_win_prob'] = 100 - results['dem_win_prob']
results['margin_low'] = round(results['margin'] - results['total_std'] * 1.96, 2)
results['margin_high'] = round(results['margin'] + results['total_std'] * 1.96, 2)

print(results[['State', 'polling_avg_margin', 'margin', 'ci_95', 'margin_low', 'margin_high', 'dem_win_prob', 'rep_win_prob']])

results.to_csv('aggregated_estimates.csv', index=False)

from datetime import timedelta

START_DATE = datetime(2026, 3, 1)
date_range = [START_DATE + timedelta(days=i) for i in range((TODAY - START_DATE).days + 1)]

records = []

for current_date in date_range:
    # Filter to polls available on this date
    available = polls[polls['Last Date'] <= current_date].copy()
    available['days_ago'] = (current_date - available['Last Date']).dt.days
    
    # Drop polls older than 120 days
    available = available[available['days_ago'] <= 120]
    
    if len(available) == 0:
        continue
    
    # Recalculate recency weight for this date
    available['w_recency'] = available['days_ago'].apply(recency_weight)
    available['weight'] = available['w_recency'] * available['w_sample'] * available['w_quality']
    
    # Run aggregation per state
    for state, group in available.groupby('State'):
        group = group.copy()
        group['weight_norm'] = group['weight'] / group['weight'].sum()
        dem = (group['D_adjusted'] * group['weight_norm']).sum()
        rep = (group['R_adjusted'] * group['weight_norm']).sum()
        margin = dem - rep
        avg_moe = (group['MOE'] * group['weight_norm']).sum()
        ci = (avg_moe / np.sqrt(len(group))) * 1.96
        
        days_to_election = (ELECTION_DAY - current_date).days
        fund_unc = 3.0 * np.sqrt(days_to_election / 365)
        polling_std = ci / 1.96
        total_std = np.sqrt(polling_std**2 + fund_unc**2)
        prob_dem = round(stats.norm.cdf(margin / total_std) * 100, 1)
        
        records.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'state': state,
            'dem_estimate': round(dem, 2),
            'rep_estimate': round(rep, 2),
            'polling_avg_margin': round(margin, 2),
            'margin': round(margin, 2),
            'ci_95': round(ci, 2),
            'dem_win_prob': prob_dem,
            'rep_win_prob': round(100 - prob_dem, 1),
            'n_polls': len(group)
        })

timeseries = pd.DataFrame(records)
timeseries.to_csv('timeseries_estimates.csv', index=False)