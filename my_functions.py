import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.ensemble import RandomForestClassifier

class HelperFunctions:
    def __init__(self):
        pass
    # Function to calculate the birthday, gender and age from birth_number
    def calculate_birthday(self, df):
        
        genders = []
        birthdays = []
        ages = []
        
        base_date = datetime(1999, 12, 31)
        
        for birth_number in df['birth_number']:
            birth_number_str = str(birth_number)
            
            year = int("19" + birth_number_str[:2])
            month = int(birth_number_str[2:4])
            day = int(birth_number_str[4:6])
            
            if month > 12:
                gender = "female"
                month -= 50  # minus 50 for females
            else:
                gender = "male"
            
            birth_day = datetime(year, month, day)
            age = base_date.year - birth_day.year - ((base_date.month, base_date.day) < (birth_day.month, birth_day.day))
            
            genders.append(gender)
            birthdays.append(birth_day)
            ages.append(age)
        
        df['gender'] = genders
        df['birth_date'] = birthdays
        df['age'] = ages
        
        df = df.drop(columns=['birth_number'])
        
        return df
    
    def get_transactions_within_window(self, trans_disp, client_id, start_date, end_date):
        client_trans = trans_disp[trans_disp['client_id'] == client_id]
        window_trans = client_trans[(client_trans['trans_date'] >= start_date) & (client_trans['trans_date'] <= end_date)]
        return window_trans
    
    def extract_features(self, transactions):
        features = {}
        features['amount_transactions'] = len(transactions)
        features['total_amount'] = transactions['trans_amount'].sum()
        features['average_amount'] = transactions['trans_amount'].mean()
        features['max_balance'] = transactions['balance'].max()
        features['min_balance'] = transactions['balance'].min()
        return pd.Series(features)
    
    def calculate_wealth_metrics(self, transactions):
        if transactions.empty:
            return pd.Series({
                'avg_balance': 0,
                'balance_std': 0,
                'ending_balance': 0
            })
        else:
            balances = transactions['balance']
            avg_balance = balances.mean()
            balance_std = balances.std()
            # Assuming transactions are sorted by date
            ending_balance = transactions.sort_values('trans_date', ascending=True)['balance'].iloc[-1]
            return pd.Series({
                'avg_balance': avg_balance,
                'balance_std': balance_std,
                'ending_balance': ending_balance
            })
        
    def calculate_transaction_metrics(self, transactions):
        if transactions.empty:
            return pd.Series({
                'num_debit_transactions': 0,
                'num_credit_transactions': 0,
                'total_debit_amount': 0,
                'total_credit_amount': 0,
                'avg_transaction_amount': 0,
                'transaction_frequency': 0
            })
        else:
            num_transactions = len(transactions)
            num_debit_transactions = len(transactions[transactions['trans_type'] == 'withdrawal'])
            num_credit_transactions = len(transactions[transactions['trans_type'] == 'credit'])
            total_debit_amount = transactions[transactions['trans_type'] == 'withdrawal']['trans_amount'].sum()
            total_credit_amount = transactions[transactions['trans_type'] == 'credit']['trans_amount'].sum()
            avg_transaction_amount = transactions['trans_amount'].mean()
            # Calculate transaction frequency as transactions per day
            num_days = (transactions['trans_date'].max() - transactions['trans_date'].min()).days + 1
            transaction_frequency = num_transactions / num_days if num_days > 0 else 0
            return pd.Series({
                'num_debit_transactions': num_debit_transactions,
                'num_credit_transactions': num_credit_transactions,
                'total_debit_amount': total_debit_amount,
                'total_credit_amount': total_credit_amount,
                'avg_transaction_amount': avg_transaction_amount,
                'transaction_frequency': transaction_frequency
            })


    # Assuming 'transactions' has columns: trans_date, trans_amount, balance
    def tsfresh_features(self, transactions):
        if transactions.empty:
            return pd.Series()

        # Ensure proper time series structure
        transactions = transactions.sort_values('trans_date')
        transactions['id'] = 0  # tsfresh needs a grouping ID
        
        # Extract features for 'trans_amount' and 'balance'
        extracted_features = extract_features(
            transactions[['id', 'trans_date', 'trans_amount', 'balance']],
            column_id='id',
            column_sort='trans_date',
            default_fc_parameters=MinimalFCParameters(),
            disable_progressbar=True
        )
        return extracted_features.iloc[0]  # Return as Series for easy joining
    
    def plot_monthly_trends(self, trans_disp, account_id):
        # Filter transactions for the specified account
        account_data = trans_disp[trans_disp['account_id'] == account_id].copy()

        # Ensure 'date' is datetime
        account_data['trans_date'] = pd.to_datetime(account_data['trans_date'], errors='coerce')

        # Extract year and month
        account_data['year_month'] = account_data['trans_date'].dt.to_period('M')

        # Aggregate monthly metrics: average balance and total amount
        monthly_stats = account_data.groupby('year_month').agg(
            avg_balance=('balance', 'mean'),
            total_amount=('trans_amount', 'sum')
        ).reset_index()

        # Convert Period to string for plotting
        monthly_stats['year_month'] = monthly_stats['year_month'].astype(str)

        # Plot Average Balance Over Time
        plt.figure(figsize=(15,5))
        plt.plot(monthly_stats['year_month'], monthly_stats['avg_balance'], marker='o')
        plt.title(f'Monthly Average Balance for Account {account_id}')
        plt.xlabel('Year-Month')
        plt.ylabel('Average Balance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Plot Total Amount (Revenue) Over Time
        plt.figure(figsize=(15,5))
        plt.plot(monthly_stats['year_month'], monthly_stats['total_amount'], marker='o')
        plt.title(f'Monthly Total Amount for Account {account_id}')
        plt.xlabel('Year-Month')
        plt.ylabel('Total Amount')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


            # Function to evaluate model with varying number of top features
    def evaluate_model_with_top_features(self, y_train_exp, importance_df, X_train_exp, X_test_exp, random_search, roc_auc_score, precision_score, recall_score, f1_score, y_test_exp, num_features):
        selected_features = importance_df['Feature'].head(num_features).tolist()
        X_train_subset = X_train_exp[selected_features]
        X_test_subset = X_test_exp[selected_features]
        
        model = RandomForestClassifier(random_state=42, **random_search.best_params_)
        model.fit(X_train_subset, y_train_exp)
        
        y_pred = model.predict(X_test_subset)
        y_pred_proba = model.predict_proba(X_test_subset)[:, 1]
        
        roc_auc = roc_auc_score(y_test_exp, y_pred_proba)
        precision = precision_score(y_test_exp, y_pred)
        recall = recall_score(y_test_exp, y_pred)
        f1 = f1_score(y_test_exp, y_pred)
        
        return {'Num_Features': num_features, 'ROC AUC': roc_auc, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

