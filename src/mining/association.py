from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

def run_apriori(df, min_support=0.05, min_threshold=0.2):
    # Chọn thêm market_segment để luật kết hợp đa dạng hơn
    cols = ['deposit_type', 'customer_type', 'market_segment', 'is_canceled']
    df_subset = pd.get_dummies(df[cols])
    
    # Chạy thuật toán Apriori
    frequent_itemsets = apriori(df_subset, min_support=min_support, use_colnames=True)
    
    # Tạo luật kết hợp
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
    
    # Lọc các luật dẫn đến việc hủy phòng (is_canceled_1)
    cancel_rules = rules[rules['consequents'].apply(lambda x: 'is_canceled_1' in str(x))]
    return cancel_rules.sort_values(by='lift', ascending=False)