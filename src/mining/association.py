import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def discretize_lead_time(df):
    """Rời rạc hóa lead_time thành 3 bins theo yêu cầu đề tài."""
    if 'lead_time' in df.columns:
        df['lead_time_cat'] = pd.cut(df['lead_time'], 
                                     bins=[0, 30, 180, 1000], 
                                     labels=['Short', 'Medium', 'Long'])
    return df

def run_apriori(df, min_support=0.05, min_threshold=0.2):
    # Đảm bảo có cột lead_time_cat
    if 'lead_time_cat' not in df.columns:
        df = discretize_lead_time(df)
        
    cols = ['lead_time_cat', 'deposit_type', 'customer_type', 'market_segment', 'is_canceled']
    df_encoded = pd.get_dummies(df[cols].copy(), columns=['lead_time_cat', 'deposit_type', 'customer_type', 'market_segment'])
    
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
    
    # Lọc luật chứa is_canceled
    cancel_rules = rules[rules['consequents'].apply(lambda x: 'is_canceled' in str(x))]
    return cancel_rules.sort_values(by='lift', ascending=False)