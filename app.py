import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Passenger Survival Prediction")
st.markdown("Predict whether a passenger is likely to survive the Titanic disaster.")

pclass = st.slider("Enter the Passenger class for the user", 1,3)
sex = st.selectbox("Enter the Passenger Gender", ['male', 'female'])
sibsp = st.slider("Enter the Passenger total no. of sibling and Spouse", 1, 8)
parch = st.slider("Enter the Passenger total no. of parents and child", 1, 6)
fare = st.number_input("Enter the fare of the Passenger")
embarked = st.selectbox("Enter the passenger station from where they started", ['Southampton','Chebourg','Queenstown'])

data = pd.DataFrame([{
    'Pclass':pclass,
    "Sex": sex,
    "SibSp":sibsp,
    "Parch":parch,
    "Fare": fare,
    "Embarked":embarked
}])

model = load_model('model.h5')

with open("label_encoder.pkl", 'rb') as file:
    label = pickle.load(file)

with open("onehot_encoder.pkl", 'rb') as file:
    onehot = pickle.load(file)

with open("scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

data['Sex'] = label.transform(data['Sex'])

embarked = onehot.transform(data[['Embarked']])

embarked = pd.DataFrame(embarked, columns=onehot.get_feature_names_out())

data = pd.concat([data.drop(columns=['Embarked']), embarked],axis=1)

data[['Pclass', 'SibSp', 'Parch', 'Fare']] = scaler.transform(data[['Pclass', 'SibSp', 'Parch', 'Fare']])

y = model.predict(data)

y = y[0][0]

def chance(y):
    if y > 0.5:
        return 'The Passenger will Survive the Journey'
    else:
        return "The Passenger won't Survive"

if st.button("Predict Survival Chance"):
    st.write("Probability of Passenger Survival Change", y)
    st.write(chance(y))