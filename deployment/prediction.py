import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

# Load All Files

with open('final_pipeline.pkl', 'rb') as file_1:
  model_pipeline = pickle.load(file_1)

model_ann = load_model('customer_churn.h5')

def run():
    with st.form(key='Churn_Customer_Prediction'):
        user_id = st.text_input('User_ID', value='')
        age = st.number_input('Age', min_value=23, max_value=65, value=23)
        gender = st.selectbox('Gender', ('Male', 'Female'), index=1)
        days_since_last_login = st.number_input('Last Login', min_value=0, max_value=26, value=0)
        avg_time_spent = st.number_input('Avg. Time Spent', min_value=0, max_value=3236, value=0)
        avg_transaction_value = st.number_input('Avg. Transaction Value', min_value=800, max_value=99915, value=29271)
        avg_frequency_login_days = st.number_input('Avg. Frequency Login Days', min_value=0, max_value=73, value=0)
        points_in_wallet = st.number_input('Points in Wallet', min_value=0, max_value=2070, value=0)
        joining_date = st.date_input("Select Join Date")
        last_visit_time = st.time_input('Last Visit Time')
        st.markdown('---')

        region_category = st.selectbox('Region Category', ('Village', 'Town', 'City'), index=1)
        membership_category = st.selectbox('Membership Category', ('No Membership', 'Basic Membership', 
                                                                   'Silver Membership', 'Premium Membership',
                                                                   'Gold Membership', 'Platinum Membership'), index=1)
        preferred_offer_types = st.selectbox('Preffered Offer', ('Without Offers', 'Credit/Debit Card Offers', 
                                                                 'Gift Vouchers/Coupons'), index=1)
        medium_of_operation = st.selectbox('Medium Ops', ('Desktop', 'Mobile', 'Both'
                                                          'Gift Vouchers/Coupons'), index=1)
        internet_option = st.selectbox('Internet Ops', ('Wi-Fi', 'Fiber_Optic', 'Mobile-Data'), index=1)
        feedback = st.selectbox('Feedback', ('Poor Website', 'Poor Customer Service', 'Poor Product Quality',
                                             'Too many ads', 'No reason specified', 'Products always in Stock',
                                             'Reasonable Price', 'Quality Customer Care', 'User Friendly Website'), index=1)
        complaint_status = st.selectbox('Complaint Status', ('No Information Available', 'Not Aplicable', 'Unsolved',
                                                             'Solved', 'Solved in Follow-up'), index=1)
        
        st.markdown('---')

        joined_through_referral = st.selectbox('Join Through Referral', ('Yes', 'No'), index=1)
        used_special_discount = st.selectbox('Use Special Discount', ('Yes', 'No'), index=1)
        offer_application_preference = st.selectbox('Offer Application Preference', ('Yes', 'No'), index=1)
        past_complaint = st.selectbox('Past Complaint', ('Yes', 'No'), index=1)

        submitted = st.form_submit_button('Predict')

    data_inf = {
    'user_id': user_id,
    'age': age,
    'gender': gender,
    'region_category': region_category,
    'membership_category': membership_category,
    'joining_date': joining_date,
    'joined_through_referral': joined_through_referral,
    'preferred_offer_types': preferred_offer_types,
    'medium_of_operation': medium_of_operation,
    'internet_option': internet_option,
    'last_visit_time': last_visit_time,
    'days_since_last_login': days_since_last_login,
    'avg_time_spent': avg_time_spent,
    'avg_transaction_value': avg_transaction_value,
    'avg_frequency_login_days': avg_frequency_login_days,
    'points_in_wallet': points_in_wallet,
    'used_special_discount': used_special_discount,
    'offer_application_preference': offer_application_preference,
    'past_complaint': past_complaint,
    'complaint_status': complaint_status,
    'feedback': feedback
    }

    data_inf = pd.DataFrame([data_inf])
    data_inf_transform = model_pipeline.transform(data_inf)

    a = st.dataframe(data_inf_transform)
    b = ''

    if len(data_inf_transform) == 0:
        b = 'Not Churn'
    else:
        # Predict using ANN: Sequential API
        y_pred_inf = model_ann.predict(data_inf_transform)
        y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)
        if y_pred_inf == 0:
            b = 'Not Churn'
        else:
            b = 'Churn'

    if submitted:
        st.write('# Prediction : ', b)

if __name__ == '__main__':
    run()