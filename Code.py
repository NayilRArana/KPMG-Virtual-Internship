# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 07:45:36 2020

@author: Nayil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from functools import reduce
import os.path
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#reads each sheet into a dataframe. Returns dataframes
def read_sheets():
    transactions = pd.read_excel("KPMG_VI_New_raw_data_update_final.xlsx", sheet_name = "Transactions")
    new_customer_list = pd.read_excel("KPMG_VI_New_raw_data_update_final.xlsx", sheet_name = "NewCustomerList")
    customer_demographic = pd.read_excel("KPMG_VI_New_raw_data_update_final.xlsx", sheet_name = "CustomerDemographic")
    customer_address = pd.read_excel("KPMG_VI_New_raw_data_update_final.xlsx", sheet_name = "CustomerAddress")
    return [transactions, new_customer_list, customer_demographic, customer_address]

#performs data cleaning tasks. Returns amended dataframes
def data_cleaning(tables):
    transactions, new_customer_list, customer_demographic, customer_address = tables[0], tables[1], tables[2], tables[3]
    
    #removes transaction records with several blank column values
    transactions = transactions[(transactions.brand.notnull())]
    
    #remove U-genders with several blank column values. Also corrects gender misspellings.
    customer_demographic = customer_demographic[(customer_demographic.gender != "U")]
    customer_demographic["gender"].replace({"F": "Female", "Femal": "Female", "M": "Male"}, inplace=True)
    
    #removes records with customer ids greater than 4000. Also corrects abbreviations of state names.
    customer_address = customer_address.drop(customer_address[(customer_address.customer_id > 4000)].index)
    customer_address["state"].replace({"NSW": "New South Wales", "QLD": "Queensland", "VIC": "Victoria"}, inplace=True)
    return [transactions, new_customer_list, customer_demographic, customer_address]

#performs feature engineering tasks. Returns joined dataframe
def feature_engineering(tables):
    transactions, new_customer_list, customer_demographic, customer_address = tables[0], tables[1], tables[2], tables[3]
    
    #creates profit column and new table grouping profits by customer id.
    transactions["profit"] = transactions["list_price"] - transactions["standard_cost"]
    profits_by_customer_id = transactions.groupby("customer_id")["profit"].sum().to_frame()
    
    #creates age column from dates of birth
    customer_demographic["age"] = np.floor((pd.datetime.today() - customer_demographic["DOB"]).dt.days/365)
    
    #creates joined table
    dfs = [customer_demographic, customer_address, profits_by_customer_id]
    df_final = reduce(lambda left,right: pd.merge(left,right,on='customer_id'), dfs)
    return df_final

#saves joined table to CSV file
def save_final_table(df_final):
    df_final.to_csv('df_final.csv', index=False)

#loads final table from CSV file
def load_final_table():
    df_final = pd.read_csv('df_final.csv')
    return df_final

#shows and saves a scatterplot of age vs profit
def age_vs_profit(df_final):
    sns.scatterplot('age', 'profit', data=df_final)
    plt.title("Age vs Profit")
    plt.show()
    plt.savefig('age_vs_profit.png')

#shows and saves a boxplot of age vs profit
def gender_vs_profit(df_final):
    sns.boxplot('gender', 'profit', data=df_final)
    plt.title("Gender vs Profit")
    plt.show()
    plt.savefig('gender_vs_profit.png')

#shows and saves a boxplot of wealth segment vs profit
def wealth_segment_vs_profit(df_final):
    df2 = pd.DataFrame({col:vals["profit"] for col, vals in df_final.groupby("wealth_segment", as_index=True)})
    meds = df2.median().sort_values()
    axes = df2[meds.index].boxplot(rot=0, return_type="axes")
    plt.xlabel("wealth_segment")
    plt.ylabel("profit")
    plt.title("Wealth Segment vs Profit")
    plt.show()
    plt.savefig('wealth_segment_vs_profit.png')

#shows and saves a boxplot of property valuation vs profit
def property_valuation_vs_profit(df_final):
    df2 = pd.DataFrame({col:vals["profit"] for col, vals in df_final.groupby("property_valuation", as_index=True)})
    meds = df2.median().sort_values()
    axes = df2[meds.index].boxplot(rot=0, return_type="axes")
    plt.xlabel("property_valuation")
    plt.ylabel("profit")
    plt.title("Property Valuation vs Profit")
    plt.show()
    plt.savefig('property_valuation_vs_profit.png')

#shows and saves a boxplot of job industry category vs profit
def job_industry_category_vs_profit(df_final):
    df2 = pd.DataFrame({col:vals["profit"] for col, vals in df_final.groupby("job_industry_category", as_index=True)})
    meds = df2.median().sort_values()
    axes = df2[meds.index].boxplot(rot=0, return_type="axes")
    plt.xticks(
        rotation=45, 
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-small'  
    )
    plt.xlabel("job_industry_category")
    plt.ylabel("profit")
    plt.title("Job Industry vs Profit")
    plt.show()
    plt.savefig('job_industry_category_vs_profit.png')

#shows and saves a boxplot of state vs profit, with the boxes being sorted by median.
def state_vs_profit(df_final):
    df2 = pd.DataFrame({col:vals["profit"] for col, vals in df_final.groupby("state", as_index=True)})
    meds = df2.median().sort_values()
    axes = df2[meds.index].boxplot(rot=0, return_type="axes")
    plt.xlabel("state")
    plt.ylabel("profit")
    plt.title("State vs Profit")
    plt.show()
    plt.savefig('state_segment_vs_profit.png')

#reads the sheets, cleans the data, performs feature engineering tasks, then saves the final table
def data_processing():
    tables = read_sheets()
    tables = data_cleaning(tables)
    save_final_table(feature_engineering(tables))

#saves all data visualisations.
def save_data_visualizations(df_final):
    age_vs_profit(df_final)
    gender_vs_profit(df_final)
    wealth_segment_vs_profit(df_final)
    property_valuation_vs_profit(df_final)
    job_industry_category_vs_profit(df_final)
    state_vs_profit(df_final)
    
#splits data into training and testing data, fits the model, tests it on testing data, and then outputs metrics.
def linear_modelling(df_final):
    #print(df_final.isnull().any())
    X = df_final[['gender', 'age', 'job_industry_category', 'wealth_segment', 'property_valuation', 'state']]
    y = df_final['profit']
    X = pd.get_dummies(data=X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
    print(coeff_df)
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(df.head(25))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R_squared: ', regressor.score(X_test, y_test))

def main():
    if not os.path.exists('df_final.csv'):
        data_processing()
    df_final = load_final_table()
    save_data_visualizations(df_final):
    linear_modelling(df_final)
    

if __name__ == "__main__":
    main()