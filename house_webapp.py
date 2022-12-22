import streamlit as st
import pickle
import numpy as np

lin=pickle.load(open('lin_model.pkl','rb'))
dt = pickle.load(open('dec_tree_model.pkl','rb'))
rf= pickle.load(open('rand_for_model.pkl','rb'))
xgb = pickle.load(open('xgb_model.pkl','rb'))
ada = pickle.load(open('ada_reg_model.pkl','rb'))

st.title("House Price Prediction in Delhi Web App")
html_temp = """
    <div style="background-color:lightgreen ;padding:8px">
    <h2 style="color:black;text-align:center;">House Price Prediction in Delhi</h2>
    </div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

activities = ['Linear Regression','Decision Tree','Random Forest','AdaBoost']
option = st.sidebar.selectbox('Which regression model would you like to use?',activities)
st.subheader(option)

st.write("""###### For Status selecton select numbers from 0-1 as follows:
            0: Almost Ready, 1: Ready to move """)

st.write("""###### For Type selecton select numbers from 0-1 as follows:
            0: Appartment, 1: Builder Floor """)

st.write("""###### For Transaction selecton select numbers from 0-1 as follows:
            0: New Property, 1: Resale """)


st.write("""###### For Furnishing selecton select numbers from 0-2 as follows:
            0: Furnished, 1: Semi Furnished, 2: Unfurnished """)

status = [0,1]
status_option = st.sidebar.selectbox("Choose the Status",status)
status_option=float(status_option)

type = [0,1]
type_option = st.sidebar.selectbox("Choose the type",type)
type_option=float(type_option)

transaction = [0,1]
trans_option = st.sidebar.selectbox("Choose the transaction",transaction)
trans_option=float(trans_option)

furnish = [0,1,2]
furnish_option = st.sidebar.selectbox("Choose the type of Furnishing",furnish)
furnish_option=float(furnish_option)

area = st.slider('Select Area', 900, 11050)
bathroom = st.slider('Select Number of bathrooms', 1, 7)
BHK = st.slider('Select BHK', 1, 10)
parking = st.slider('Select Number of parkings available', 1, 5)
price_per_sqft = st.slider('Select price_per_sqft', 1200.0, 150000.0)
locality = st.slider('Select locality', 0, 365)


inputs=[[area,BHK,bathroom,furnish_option,locality,parking,status_option,trans_option,type_option,
price_per_sqft]]


if st.button('Predict'):
    if option=='Linear Regression':
        st.success(lin.predict(inputs))
    elif option=='Decision Tree':
        st.success(dt.predict(inputs))
    elif option=='Random Forest':
        st.success(rf.predict(inputs))
    # elif option=='XGBoost':
    #     st.success(xgb.predict(inputs))
    else:
        st.success(ada.predict(inputs))
