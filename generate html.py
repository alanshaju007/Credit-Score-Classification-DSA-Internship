columns = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
           'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
           'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
           'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
           'Credit_History_Age', 'Total_EMI_per_month', 'Amount_invested_monthly',
           'Monthly_Balance', 'Occupation_Accountant', 'Occupation_Architect',
           'Occupation_Developer', 'Occupation_Doctor', 'Occupation_Engineer',
           'Occupation_Entrepreneur', 'Occupation_Journalist', 'Occupation_Lawyer',
           'Occupation_Manager', 'Occupation_Mechanic', 'Occupation_Media_Manager',
           'Occupation_Musician', 'Occupation_Scientist', 'Occupation_Teacher',
           'Occupation_Writer',
           'Payment_Behaviour_High_spent_Large_value_payments',
           'Payment_Behaviour_High_spent_Medium_value_payments',
           'Payment_Behaviour_High_spent_Small_value_payments',
           'Payment_Behaviour_Low_spent_Large_value_payments',
           'Payment_Behaviour_Low_spent_Medium_value_payments',
           'Payment_Behaviour_Low_spent_Small_value_payments',
           'Type_of_Loan_encoded', 'Credit_Mix_encoded',
           'Payment_of_Min_Amount_encoded', 'Quarter_1', 'Quarter_2', 'Quarter_3']

html_code = ""
for column in columns:
    html_code += f'<label for="{column}">{column}</label>\n'
    html_code += f'<input type="number" name="{column}" required><br><br>\n'

print(html_code)
