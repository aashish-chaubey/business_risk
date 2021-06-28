import pandas as pd
import numpy as np

def prepare_input(data):
    data = data.drop(labels=['agent_id'], axis=1)
    revenue_df = data.filter(regex='gtv')
    revenue_month_df = data.filter(regex='Months')
    revenue_week_df = data.filter(regex='weeks')
    revenue_day_df = data.filter(regex='days')
    revenue_data = {
        'all_gtv_last12Months_mean': data.mean(axis=1),
        'all_gtv_last10weeks_mean': data.mean(axis=1),
        'all_gtv_last10days_mean': data.mean(axis=1)
    }
    data = data.drop(labels=revenue_df.columns.values, axis=1)
    data = pd.concat([data, pd.DataFrame(revenue_data)], axis=1)
    
    growth_norm_month_df = data.filter(regex='growth_m')
    growth_mean = pd.Series(growth_norm_month_df.mean(axis=1), name="all_norm_growth_mean")
    data = data.drop(labels=growth_norm_month_df.columns.values, axis=1)
    data = pd.concat([data, growth_mean], axis=1)
    
    data = data.drop(labels=data.filter(regex='3M_').columns.values, axis=1)
    
    gap_df = data.filter(regex='gap')
    gap_mean = pd.Series(gap_df.mean(axis=1), name="gaps_mean")
    data = data.drop(labels=gap_df.columns.values, axis=1)
    data = pd.concat([data, gap_mean], axis=1)
    
    all_last_df = data.filter(regex='all_last')
    all_last_mean = pd.Series(all_last_df.mean(axis=1), name="all_last_mean")
    data = data.drop(labels=all_last_df.columns.values, axis=1)
    data = pd.concat([data, all_last_mean], axis=1)
    
    all_ystrday_df = data.filter(regex='all_ystrday')
    all_ystrday_mean = pd.Series(all_ystrday_df.mean(axis=1), name="all_ystrday_mean")
    data = data.drop(labels=all_ystrday_df.columns.values, axis=1)
    data = pd.concat([data, all_ystrday_mean], axis=1)
    
    all_mrr_df = data.filter(regex='all_mrr')
    all_mrr_mean = pd.Series(all_mrr_df.mean(axis=1), name="all_mrr_mean")
    data = data.drop(labels=all_mrr_df.columns.values, axis=1)
    data = pd.concat([data, all_mrr_mean], axis=1)
    
    data = data.drop(labels=['all_seg', 'all_risk_cm', 
                             'all_consistency_index', 'all_growth_index', 
                             'all_norm_growth_index_last', 'all_7days_max_thisvs10w',
                             'all_7days_vslast_month7days', 'all_7days_vslast7days', 
                             'all_7days_mean_thisvs4w', 'all_7days_min_thisvs4w', 
                             'all_7days_trend_vs10weeks', 'all_lst30days_vsmin_lst3m', 
                             'all_mtd_vs_mean_lst3M', 'all_mtd_vs_min_lst3M'], axis=1)
    
    return data