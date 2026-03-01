from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

def run_apriori(df):
    # Chọn một số cột quan trọng và chuyển về dạng Categorical
    cols = ['deposit_type', 'customer_type', 'is_canceled']
    df_subset = pd.get_dummies(df[cols])
    
    # Chạy thuật toán Apriori
    frequent_itemsets = apriori(df_subset, min_support=0.05, use_colnames=True)
    
    # Tạo luật kết hợp
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
    
    # Lọc các luật dẫn đến việc hủy phòng (is_canceled_1)
    cancel_rules = rules[rules['consequents'].apply(lambda x: 'is_canceled_1' in str(x))]
    return cancel_rules.sort_values(by='lift', ascending=False)