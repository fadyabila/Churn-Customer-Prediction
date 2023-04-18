import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

# Melebarkan visualisasi untuk memaksimalkan browser
st.set_page_config(
    page_title='Churn Customer',
    layout='wide',
    initial_sidebar_state='expanded'
)

def run():
    # Membuat title
    st.title('Churn Customer Prediction')
    st.write('### by Fadya Ulya Salsabila')

    # Menambahkan Gambar
    image = Image.open('churn.png')
    st.image(image, caption='Illustration of Churn Customer')

    # Menambahkan Deskripsi
    st.write('## Background')
    st.write("""
    A make-up company "Sister" wants to minimize the risk of a customer stopping using their product. 
    The company then analyzes the history of its customers in making purchases based on time and frequency. Then, this company also looks at the feedback that customers have given it.
    This is intended to determine customer predictions whether to stop using their product or not. 
    Because if many customers stop, the company will evaluate product sales and marketing to customers. In addition, the company will also provide discounts and special offers to loyal customers.
    
    The objectives from this analysis and modeling in this dataset are:
    1. Find out the customer prediction, whether customer churn or not.
    2. Find out the best model prediction using Artificial Neural Network (ANN).""")

    st.write('## Dataset')
    st.write("""
    The dataset is from Github Milestones 1 Hacktiv8 `churn.csv` that contains 22 columns.
    1. `user_id`:	ID of a customer
    2. `age`:	Age of a customer
    3. `gender`:	Gender of a customer
    4. `region_category`:	Region that a customer belongs to
    5. `membership_category`:	Category of the membership that a customer is using
    6. `joining_date`:	Date when a customer became a member
    7. `joined_through_referral`:	Whether a customer joined using any referral code or ID
    8. `preferred_offer_types`:	Type of offer that a customer prefers
    9. `medium_of_operation`:	Medium of operation that a customer uses for transactions
    10. `internet_option`:	Type of internet service a customer uses
    11. `last_visit_time`:	The last time a customer visited the website
    12. `days_since_last_login`:	Number of days since a customer last logged into the website
    13. `avg_time_spent`:	Average time spent by a customer on the website
    14. `avg_transaction_value`:	Average transaction value of a customer
    15. `avg_frequency_login_days`:	Number of times a customer has logged in to the website
    16. `points_in_wallet`:	Points awarded to a customer on each transaction
    17. `used_special_discount`:	Whether a customer uses special discounts offered
    18. `offer_application_preference`:	Whether a customer prefers offers
    19. `past_complaint`:	Whether a customer has raised any complaints
    20. `complaint_status`:	Whether the complaints raised by a customer was resolved
    21. `feedback`:	Feedback provided by a customer
    22. `churn_risk_score`:	Churn score (0 : Not churn, 1 : Churn)""")

    # Membuat Garis Lurus
    st.markdown('---')

    # Membuat Sub Headrer
    st.subheader('EDA for Churn Customer')

    # Magic Syntax
    st.write(
    ' On this page, the author will do a simple exploration.'
    ' The dataset used is the Churn Customer dataset.'
    ' This dataset comes from Github Project Hacktiv8.')

    # Show DataFrame
    df1 = pd.read_csv('churn.csv')
    st.dataframe(df1)

    # Membuat Barplot
    st.write('#### Churn Risk Plot')
    fig = plt.figure(figsize=(10,7))
    sns.countplot(x='churn_risk_score', data=df1, palette="PuRd")
    st.pyplot(fig)
    st.write('The target data is balanced.')

    st.write('#### Gender Based on Churn Risk')
    fig1, ax1 = plt.subplots(figsize=(15, 8))
    sns.countplot(x='gender', hue='churn_risk_score', data=df1, ax=ax1)
    st.pyplot(fig1)
    st.write('Gender distribution is normal between men and women.')

    # Mengelompokkan Usia
    bins = [8, 20, 30, 40, 50, 60, 120]
    labels = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69']
    df1['agerange'] = pd.cut(df1.age, bins, labels = labels,include_lowest = True)

    # Menampilkan visualisasi usia berdasarkan churn risk
    st.write('#### Age Based on Churn Risk')
    fig2, ax2 = plt.subplots(figsize=(10,7))
    sns.countplot(x='agerange', data=df1, hue="churn_risk_score", ax=ax2)
    st.pyplot(fig2)
    st.write('Customers in this company varies greatly, ranging from 10-64 years old. ')

    # Membuat heatmap correlation
    st.write('#### Heatmap Correlation')
    fig = plt.figure(figsize = (15,8))
    sns.heatmap(df1.corr(), annot = True, square = True)
    st.pyplot(fig)
    st.write("""
    The heatmap correlation above shows that the column that has a very high relationship with churn risk is the `avg_freqeuncy_login_days` column with score `0.11`. This column shows how many customers log in in a day. 
    It means they are still interested in the product in this company. Meanwhile, `avg_transaction_value` have a strong negative correlation with churn risk witn score `-0.22`. 
    This shows that the number of purchase transactions on this product has no significant effect on customer churn.""")

    # Membuat internet option berdasarkan churn risk
    st.write('#### Internet Option Based on Churn Risk')
    fig3, ax3 = plt.subplots(figsize=(10,7))
    sns.countplot(x='internet_option', data=df1, hue="churn_risk_score", ax=ax3, palette="Blues")
    st.pyplot(fig3)
    st.write("""
    Bar plot visualization above, shows that the `internet option` of customers doesn't have a strong correlation with churn risk. 
    Distribution of internet option data almost have the same number of values and there is no significant difference. 
    Customers who use the internet with Wi-Fi, Fiber Optic, and Mobile Data are almost the same.""")

    # Membuat region category berdasarkan churn risk
    st.write('#### Region Category Based on Churn Risk')
    fig4, ax4 = plt.subplots(figsize=(10,7))
    sns.countplot(x='region_category', data=df1, hue="churn_risk_score", ax=ax4, palette="Blues")
    st.pyplot(fig4)
    st.write("""
    Based on customer region, there is no significant correlation with churn risk. 
    It's just that many customers of this product live in town areas compared to villages and cities.""")

    # Membuat membership category berdasarkan churn risk
    st.write('#### Membership Category on Churn Risk')
    fig5, ax5 = plt.subplots(figsize=(10,7))
    sns.countplot(y='membership_category', data=df1, hue="churn_risk_score", ax=ax5, palette="Blues")
    st.pyplot(fig5)
    st.write("""
    In `membership_category` column, customers which include in `No Membership` dan `Basic Membership` are customers with the highest churn risk. 
    This can happen because the customer is deemed not a loyal customer so the risk of stopping the transaction is high. 
    In contrast to silver, premium, gold, and platinum members where customers are considered loyal to product transactions.
    """)

    # Membuat Histogram Berdasarkan Input User
    st.write('#### Histogram Based On User Input')
    pilihan = st.selectbox('Choose Column : ', ('age', 'gender', 'days_since_last_login', 'avg_time_spent', 
                                                'avg_transaction_value', 'avg_frequency_login_days', 
                                                'points_in_wallet'))
    fig = plt.figure(figsize=(15,5))
    sns.histplot(df1[pilihan], bins=30, kde=True)
    st.pyplot(fig)

if __name__ == '__main__':
    run()