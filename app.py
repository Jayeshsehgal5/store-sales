import pandas as pd
import streamlit as st
from PIL import Image
import sklearn
import joblib
import Preprocessing
import Model_creation
df=pd.read_csv("storesales_train.csv")
data=Preprocessing.Preprocessing(df)
X,Y=data.preprocessing()
MODEL=Model_creation.model_creation(X,Y)
MODEL.model_creation()
model=joblib.load("model_save")
def Predict(Item_Weight, Item_Visibility,Item_MRP,Item_Fat_Content,
       Outlet_Size,Outlet_Location_Type,Outlet_Type):
    x="a"
    x1="b"
    x2="c"
    x3="d"
    x4="f"
    x5="j"
    #Conversion to binary(Item_Fat_Content)
    if Item_Fat_Content=="Low Fat":
        x=1
    else:
        x=0
    #Conversion to Labels(Outlet_Size)
    if Outlet_Size=="High":
        x1=3
    elif Outlet_Size=="Medium":
        x1=2
    else:
        x1=1
    #Outlet_Location_Type
    if Outlet_Location_Type == "Tier 1":
        x2 = 1
    elif Outlet_Location_Type== "Tier 2":
        x2 = 2
    else:
        x2 = 3
    #Outlet_Type
    if Outlet_Type == "SupermarketType1":
        x3 = 1
        x4 = 0
        x5 = 0
    elif Outlet_Type == "SupermarketType2":
        x3 = 0
        x4 = 1
        x5 = 0
    elif Outlet_Type=="SupermarketType3":
        x3 = 0
        x4 = 0
        x5 = 1
    else:
        x3=0
        x4=0
        x5=0
    pred1 = model.predict([[Item_Weight, Item_Visibility,Item_MRP,x,
       x1,x2,x3,x4,x5]])
    final_predict=10**pred1
    return(final_predict)
def start1():
    img1 = Image.open('slide_1.jpg')
    img1 = img1.resize((500, 300))   
    html_temp = """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Store Sales Prediction ML App </h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    Item_Weight = st.number_input("Item_Weight in kg")
    Item_Visibility = st.number_input("Item_Visibility")
    Item_MRP = st.number_input("Item_MRP")
    Item_Fat_Content = st.radio("Select Fat_Content ",("Low Fat","Regular"))
    Outlet_Size = st.radio("Select outlet Size: ", ('High', 'Medium', 'Small'))
    Outlet_Location_Type=st.radio("Select outlet location_type: ", ('Tier 1', 'Tier 2', 'Tier 3'))
    Outlet_Type= st.radio("Select outlet type: ", ('SupermarketType1', 'SupermarketType2','SupermarketType3',"Grocery Store"))
    if st.button("Predict"):
        st.success('Total sales Prediction is Rs{}'.format(Predict(Item_Weight, Item_Visibility,Item_MRP,Item_Fat_Content,Outlet_Size,Outlet_Location_Type,Outlet_Type)))
start1()
