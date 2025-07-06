import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('Machine_Learning_Projects/Admission_predictor_project/jamboree_rf_model.pkl')
scaler = joblib.load('Machine_Learning_Projects/Admission_predictor_project/jamboree_scaler.pkl')

st.title('Admission predictor')

st.write('This app predicts **Chance of admit** based on student profile inputs')

# input fields

toefl = st.number_input('TOEFL score', min_value = 0, max_value = 120, value = 100)
gre = st.number_input('GRE score', min_value = 0, max_value = 340, value = 300)
cgpa = st.number_input('CGPA' , min_value = 0.0, max_value = 10.0, value = 8.0)
sop = st.slider('SOP strength', min_value = 1.0, max_value = 5.0, value = 3.0)
lor = st.slider('LOR strength', min_value = 1.0, max_value = 5.0, value = 3.0)
uni_rating = st.slider('University Rating', min_value = 1.0, max_value = 5.0, value = 3.0)
research = st.selectbox('Research Experience', ['Yes', 'No'])

#encode research
research_encoded = 1 if research == 'Yes' else 0

#input for model
input_data = np.array([[gre, toefl, uni_rating, sop, lor, cgpa, research_encoded]])

#scale input data
input_scaled = scaler.transform(input_data)

#predict button
if st.button('predict chance of admit'):
    prediction_original_scale = np.exp(model.predict(input_scaled)[0])
    st.success(f'predicted chance of admit: {round(prediction_original_scale * 100),2}')


feature_names = ['gre_score', 'toefl_score', 'university_rating', 'sop', 'lor', 'cgpa', 'research']
importances = [0.026671, 0.018226, 0.002940, 0.001788, 0.015866, 0.067581, 0.011940]

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)


st.write("### 🔍 Feature Importance in Model")
st.dataframe(importance_df)


import altair as alt

chart = alt.Chart(importance_df).mark_bar().encode(
    x=alt.X('Importance', scale=alt.Scale(domain=[0, max(importances)+0.01])),
    y=alt.Y('Feature', sort='-x')
).properties(title="Feature Importance")

st.altair_chart(chart, use_container_width=True)











