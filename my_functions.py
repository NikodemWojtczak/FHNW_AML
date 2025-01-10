from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
from matplotlib.ticker import ScalarFormatter
from scipy.stats import linregress
from sklearn.metrics import roc_curve


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


    def tsfresh_features(self, transactions):
        if transactions.empty:
            return pd.Series()

        transactions = transactions.sort_values('trans_date')
        transactions['id'] = 0  # tsfresh needs a grouping ID
        
        extracted_features = extract_features(
            transactions[['id', 'trans_date', 'trans_amount', 'balance']],
            column_id='id',
            column_sort='trans_date',
            default_fc_parameters=MinimalFCParameters(),
            disable_progressbar=True
        )
        return extracted_features.iloc[0]  # Return as Series for easy joining
    
    def plot_monthly_trends(self, trans_disp, account_id):

        account_data = trans_disp[trans_disp['account_id'] == account_id].copy()

        account_data['trans_date'] = pd.to_datetime(account_data['trans_date'], errors='coerce')

        account_data['year_month'] = account_data['trans_date'].dt.to_period('M')

        monthly_stats = account_data.groupby('year_month').agg(
            avg_balance=('balance', 'mean'),
            total_amount=('trans_amount', 'sum')
        ).reset_index()

        monthly_stats['year_month'] = monthly_stats['year_month'].astype(str)

        # Plot Average Balance Over Time
        plt.figure(figsize=(15,5))
        plt.plot(monthly_stats['year_month'], monthly_stats['avg_balance'], marker = 'o', color = 'forestgreen')
        plt.title(f'Monthly Average Balance for Account {account_id}')
        plt.xlabel('Year-Month')
        plt.ylabel('Average Balance')
        plt.xticks(rotation = 45)
        plt.tight_layout()
        plt.show()

        # Plot Total Amount Over Time
        plt.figure(figsize=(15,5))
        plt.plot(monthly_stats['year_month'], monthly_stats['total_amount'], marker = 'o', color = 'forestgreen')
        plt.title(f'Monthly Total Amount for Account {account_id}')
        plt.xlabel('Year-Month')
        plt.ylabel('Total Amount')
        plt.xticks(rotation = 45)
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

    def plot_distribiution_of_age(self, client):
        client['birth_year'] = pd.to_datetime(client['birth_date']).dt.year
        client_count_per_year = client.groupby('birth_year')['client_id'].nunique()

        # Plot
        plt.figure(figsize=(16, 6))
        client_count_per_year.plot(kind='bar', color = 'forestgreen')
        plt.title('How many Clients are born in what Year')
        plt.xlabel('Year of Birth')
        plt.ylabel('Amount of Unique Clients')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10,6))
        sns.histplot(client['age'], bins=20, kde=True, color = 'forestgreen')
        plt.title('Age Distribution of Clients')
        plt.xlabel('Age')
        plt.ylabel('Amount of Clients')
        plt.show()
    
    def plot_gender_distribiution(self, client):
        colors = ['forestgreen', 'lightpink']
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        gender_counts = client['gender'].value_counts()

        axes[0].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=colors)
        axes[0].set_title('Distribution of Gender')

        axes[1].boxplot([client[client['gender'] == 'male']['age'], client[client['gender'] == 'female']['age']],
                        labels=['Men', 'Women'], patch_artist=True,
                        boxprops=dict(facecolor=colors[0]),  
                        medianprops=dict(color='black'),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'))
        axes[1].set_ylabel('Age')
        axes[1].set_title('Age Distribution with Genders')

        # Plotting
        plt.tight_layout()
        plt.show()

    def plot_average_salary_distribiution(self, district):
        district.sort_values('average_salary', inplace=True)
        plt.figure(figsize=(12, 6))
        plt.bar(district['district_name'], district['average_salary'], color='forestgreen')
        plt.xlabel("District")
        plt.ylabel("Average Salary")
        plt.title("Average Salary in Each District")
        plt.xticks(rotation=90)
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.scatter(district['inhabitants'], district['average_salary'], color='forestgreen')
        plt.xlabel("Number of Inhabitants")
        plt.ylabel("Average Salary")
        plt.title("Average Salary Distribution by Number of Inhabitants")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        plt.ticklabel_format(style='plain', axis='x')  
        plt.show()


        # The same plot but with a regression line
        x = district['inhabitants']
        y = district['average_salary']
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        regression_line = slope * x + intercept

        # Plot
        plt.figure(figsize=(12, 6))
        plt.scatter(x, y, color='forestgreen', label='Data Points')
        plt.plot(x, regression_line, color='red', label=f'Regression Line (R² = {r_value**2:.2f})')
        plt.xlabel("Number of Inhabitants")
        plt.ylabel("Average Salary")
        plt.title("Average Salary Distribution by Number of Inhabitants")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        plt.ticklabel_format(style='plain', axis='x')
        plt.legend()
        plt.show()

    def plot_hostogram_residuals(self, residuals):
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=20, color='green', edgecolor='black', alpha=0.7)
        plt.axvline(0, color='red', linestyle='--', label='Zero Line')
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Histogram of Residuals")
        plt.legend()
        plt.show()

    def plot_disposition_type(self, disp):
        sns.countplot(data=disp, x='type', color = 'forestgreen')
        plt.title('Disposition Types')
        plt.xlabel('Type')
        plt.ylabel('Count')
        plt.show()

    def plot_transaction_info(self, trans):
        # Visualising the transaction types 
        sns.countplot(data=trans, x='type', color = 'forestgreen')
        plt.title('Transaction Types')
        plt.xlabel('Type')
        plt.ylabel('Count')
        plt.show()

        # Visualising the transaction operations
        plt.figure(figsize=(12,6))
        sns.countplot(data=trans, y='operation', order=trans['operation'].value_counts().index, color = 'forestgreen')
        plt.title('Transaction Operations')
        plt.xlabel('Count')
        plt.ylabel('Operation')
        plt.show()

        # How many transactions do the clients have in average?
        transaction_counts = trans.groupby('transaction_type')['trans_id'].count().reset_index(name='transaction_count')

        # Plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
        axes[0, 0].boxplot(transaction_counts['transaction_count'], patch_artist=True, boxprops=dict(facecolor='forestgreen', color='black'))
        axes[0, 0].set_title("Boxplot of Transaction Counts")
        axes[0, 0].set_ylabel("Number of Transactions")
        axes[0, 0].set_xticks([]) 
        bars = axes[0, 1].bar(transaction_counts['transaction_type'], transaction_counts['transaction_count'], color='forestgreen')
        axes[0, 1].set_title("Barplot of Transaction Counts by Type")
        axes[0, 1].set_xlabel("Transaction Type")
        axes[0, 1].set_ylabel("Number of Transactions")
        axes[0, 1].tick_params(axis='x', rotation=45)  

        for bar in bars:
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            str(bar.get_height()), ha='center', fontsize=10)
        for idx, transaction_type in enumerate(transaction_counts['transaction_type']):
            subset = trans[trans['transaction_type'] == transaction_type]
            axes[1, 0].boxplot(subset['trans_id'], positions=[idx], patch_artist=True, 
                                boxprops=dict(facecolor='forestgreen', color='black'))
        axes[1, 0].set_title("Boxplots by Transaction Type")
        axes[1, 0].set_xticks(range(len(transaction_counts['transaction_type'])))
        axes[1, 0].set_xticklabels(transaction_counts['transaction_type'], rotation=45)
        axes[1, 0].set_ylabel("Transaction IDs")
        axes[1, 1].pie(
            transaction_counts['transaction_count'],
            labels=transaction_counts['transaction_type'],
            autopct='%1.1f%%',
            startangle=140,
            colors=['forestgreen', 'salmon', 'skyblue', 'maroon', 'lightgreen', 'yellow', 'orange'] * len(transaction_counts)
        )
        axes[1, 1].set_title("Percentage of Transaction Types")

        plt.tight_layout()
        plt.show()

        # See how much money is being transacted in each transaction
        plt.figure(figsize=(10,6))
        sns.histplot(trans['amount'], bins=100, kde=True, color = 'forestgreen')
        plt.title('Transaction Amount Distribution') # potrzebny inny tytul
        plt.xlabel('Amount')
        plt.ylabel('Frequency')
        plt.show()

        # When was the first transaction meaning when did the clients open their account?
        trans['first_transaction_year_month'] = trans['date'].dt.to_period('M')
        accounts_per_month = trans.groupby('first_transaction_year_month')['account_id'].nunique()

        # Plot
        plt.figure(figsize=(10, 6))
        accounts_per_month.plot(kind='bar', color='forestgreen', alpha=0.7)
        plt.title('Number of Accounts per Month')
        plt.xlabel('Year and Month')
        plt.ylabel('Count of Unique Accounts')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


    def plot_loan_info(seld, loan):
        plt.figure(figsize=(12,6))
        sns.countplot(data=loan, x='status', color = 'forestgreen')
        plt.title('Distribution of the Loan Status')
        plt.xlabel('Status')
        plt.ylabel('Count')
        plt.show()

        # Checking how much the loans are and how the distribution looks
        plt.figure(figsize=(10,6))
        sns.histplot(loan['amount'], bins=30, kde=True, color = 'forestgreen')
        plt.title('Distribution of the Loan Amounts')
        plt.xlabel('Amount')
        plt.ylabel('Frequency')
        plt.show()

        # How popular is which duration of the loan?
        duration_counts_sorted = loan['duration'].value_counts().sort_index()

        # Plot
        plt.figure(figsize=(8, 6))
        plt.bar(duration_counts_sorted.index.astype(str), duration_counts_sorted.values, color="forestgreen")
        plt.xlabel('Loan Duration in Months')
        plt.ylabel('Frequency')
        plt.title('Frequency of Loan Durations')
        plt.xticks(rotation=0)
        plt.show()

        # How many loans are there taken per month?

        loan['loan_taken_per_month'] = loan['date'].dt.to_period('M')
        loans_per_month = loan.groupby('loan_taken_per_month').size()

        # Plot 
        plt.figure(figsize=(10, 6))
        loans_per_month.plot(kind='bar', color='forestgreen', alpha=0.7)
        plt.title('Amount of Loans taken per Month')
        plt.xlabel('Year-Month')
        plt.ylabel('Amount of Loans')
        plt.tight_layout()
        plt.show()

    def plot_card_issued_per_month(seld, card):
        card['card_year_month'] = card['issued'].dt.to_period('M')
        cards_per_month = card.groupby('card_year_month').size()

        # Plot
        plt.figure(figsize=(10, 6))
        cards_per_month.plot(kind='bar', color='forestgreen', alpha=0.7)
        plt.title('Amount of Credit Cards Issued per Month')
        plt.xlabel('Year-Month')
        plt.ylabel(' ')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_correlation_new_accounts_cards(self, card_clients):
        loans_per_month = card_clients.groupby('loan_taken_per_month')['account_id'].nunique()
        cards_per_month = card_clients.groupby('card_year_month').size()
        common_index = loans_per_month.index.intersection(cards_per_month.index)
        accounts_aligned = loans_per_month.reindex(common_index)
        cards_aligned = cards_per_month.reindex(common_index)
        correlation = accounts_aligned.corr(cards_aligned)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(accounts_aligned, cards_aligned, color='forestgreen', alpha=0.7, label='Data Points')
        m, b = np.polyfit(accounts_aligned, cards_aligned, 1)  
        plt.plot(accounts_aligned, m * accounts_aligned + b, color='orange', label='Regression Line')
        plt.xlabel('Amount of new Accounts per Month')
        plt.ylabel('Amount of newly issued Credit Cards per Month')
        plt.title(f'Correlation between new Accounts and new Credit Cards')
        plt.legend()
        plt.show()

        print(f"Correlation: {correlation:.2f}")


    def plot_roc_curves(self, y_test, y_pred_proba, y_test_exp, y_pred_proba_exp, y_pred_proba_best_rf, y_pred_proba_rf, opt_rf_metrics, rf_metrics, baseline_metrics, expanded_metrics):
        plt.figure(figsize=(10, 8))

        # Baseline Logistic Regression
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label='Baseline Logistic Regression (AUC = {:.2f})'.format(baseline_metrics['ROC AUC']))

        # Expanded Logistic Regression
        fpr_exp, tpr_exp, _ = roc_curve(y_test_exp, y_pred_proba_exp)
        plt.plot(fpr_exp, tpr_exp, label='Expanded Logistic Regression (AUC = {:.2f})'.format(expanded_metrics['ROC AUC']))

        # Random Forest
        fpr_rf, tpr_rf, _ = roc_curve(y_test_exp, y_pred_proba_rf)
        plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = {:.2f})'.format(rf_metrics['ROC AUC']))

        # Optimized Random Forest
        fpr_opt_rf, tpr_opt_rf, _ = roc_curve(y_test_exp, y_pred_proba_best_rf)
        plt.plot(fpr_opt_rf, tpr_opt_rf, label='Optimized Random Forest (AUC = {:.2f})'.format(opt_rf_metrics['ROC AUC']))

        # Plot settings
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Candidate Models')
        plt.legend(loc='lower right')
        plt.show()


    def plot_roc_curves_c(y_test, **pred_proba_vars):
        """
        Plot ROC curves for multiple models.

        Parameters:
        - y_test: True labels for the test dataset.
        - pred_proba_vars: Keyword arguments where keys are model names and values are predicted probabilities.
        """
        plt.figure(figsize=(12, 10))

        # Iterate over the predicted probabilities
        for model_name, y_pred_proba in pred_proba_vars.items():
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_value = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_value:.2f})")

        # Plot settings
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')  # Diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Candidate Models')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
