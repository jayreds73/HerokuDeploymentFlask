#from flask import Flask,request
import pandas as pd
import pickle as pkl
import streamlit as st


# Run this - "streamlit run filename" in command prompt

# load the model at the start of the app
pickle_in = open('model.pkl','rb')
model_iris = pkl.load(pickle_in)

def get_description(int_code):
    if (int_code==0):
        desc = 'Setosa'
    elif (int_code == 1):
        desc = 'Versicolour'
    else:
        desc = 'Virginica'
    return desc


def Welcome():
    return "Hello world, Jayanth"


def iris_predict(sl,sw,pl,pw):
    prediction = model_iris.predict([[sl,sw,pl,pw]])
    return "The prediction is: " + get_description(int(prediction[0]))


def main():
    #Gives Title
    st.title("Iris Data Set Prediction")

    # Creates look and feel -- see more for html
    html_temp = """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Streamlit Rendered App for IRIS prediction </h2>
        </div>
        """
    # Executes HTML
    st.markdown(html_temp, unsafe_allow_html=True)

    sl = float(st.text_input('Sepal Length','1.25'))
    sw = float(st.text_input('Sepal Width','2.25'))
    pl = float(st.text_input('Petal Length','3.25'))
    pw = float(st.text_input('Petal Width','4.8'))


    prediction = ""
    # create button
    if st.button("Predict"):
        prediction = iris_predict(sl,sw,pl,pw)
    st.success(prediction)

#   prediction_t = ""
#   if st.button("Test"):
#      prediction_t = 'Pass'
#    st.success(prediction_t)

#    if st.button("About"):
#        st.text("Lets LEarn")
#        st.text("Built with Streamlit")

if(__name__=='__main__'):
    main()