import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import pandas as pd

st.title("Passenger Survival Chance In Titanic Journey")

pclass = st.slider('Enetr The Passenger Class',1,3)
sex = st.selectbox('Gender',['male','female'])
sibsp = st.slider('Enter Passenegr total nos. of Sibling and Spoce',1,8)
parch = st.slider('Enter Passenegr total nos. of Parent and Children',1,6)
fare = st.number_input('Enter the fare of passenger')
embarked = st.selectbox('Select Boarding Point',['Sounthampton','Chebourg','Queenstown'])

with open('label_encoder.pkl','rb') as file:
    label=pickle.load(file)
with open('onehot_encoder.pkl','rb') as file:
    onehot=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scale=pickle.load(file)
model = load_model('model.h5')

user_input = {'Pclass':pclass,'Sex':sex,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Embarked':embarked}
user = pd.DataFrame([user_input])

def predict(user):
    user['Sex']=label.transform(user['Sex'])
    embarked = onehot.transform(user[['Embarked']])
    embarked = pd.DataFrame(embarked,columns=onehot.get_feature_names_out())
    user = pd.concat([user.drop(columns=['Embarked']),embarked],axis=1)
    user[['Pclass','SibSp','Parch','Fare']]=scale.transform(user[['Pclass','SibSp','Parch','Fare']])
    y = model.predict(user)
    # predicted_classes = (y > 0.5).astype("int32")
    # print(f'Probablity of survival: {round(y[0][0]*100,2)} %')
    return y

if st.button("Predict Survival Chance"):
    predicted_prob = predict(user)
    st.write(user)
    st.write('Probability of Survival: ',predicted_prob[0][0])
    if predicted_prob[0][0] > 0.5:
        st.write("Surveived")
    else:
        st.write("Not Surveived")