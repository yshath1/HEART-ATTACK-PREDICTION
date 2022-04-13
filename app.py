import numpy as np
import pickle

import streamlit as st

loaded_model = pickle.load(open('C:/Users/Muthiah/Downloads/trainedfinalheartmodel1.sav', 'rb'))


def heart(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_reshape = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_reshape)

    if (prediction[0] == 0):
        return st.success('This person has less chance of heart attack')
    else:
        return st.error('This person has more chance of heart attack')


def main():
    st.title('HEART DISEASE PREDICTION')

    age = st.text_input('AGE')

    sex = st.radio("Select Gender: ", ('1', '0'))
    if (sex == '1'):
        st.info("Male")
    else:
        st.info("Female")
    st.write('<style>div.row-widget.stRadio>div{flex-direction:row;}</style>', unsafe_allow_html=True)

    chestpaintype = st.radio(
        "CHEST PAIN TYPE (0 = typical angina,1 = atypical angina,2 = non — anginal pain,3= asymptotic)",
        ('0', '1', '2', '3'))

    restingbps = st.text_input('RESTING BLOOD PRESSURE')
    cholestrol = st.text_input('CHOLESTROL')

    fastingbloodsugar = st.radio("FASTING BLOOD SUGAR( > 120mg/dl : 1, else : 0)", ('1', '0'))
    restingecg = st.radio("RESTING ECG(0-normal, 1-abnormal)", ('0', '1'))
    maxheartrate = st.text_input('MAXIMUM HEART RATE ACHIEVED')
    exerciseangina = st.radio("EXERCISE INDUCED ANGINA(1-yes, 0-no)", ('1', '0'))
    oldpeak = st.text_input('ST DEPRESSION INDUCED BY EXERCISE REALTIVE TO REST')
    STslope = st.radio("PEAK EXERCISE ST SEGMENT(0 = upsloping,1 = flat,2= downsloping)", ('0', '1', '2'))

    diagnosis = ''

    if st.button('PREDICT'):
        diagnosis = heart([age, sex, chestpaintype, restingbps, cholestrol, fastingbloodsugar, restingecg, maxheartrate,
                           exerciseangina, oldpeak, STslope])


if __name__ == '__main__':
    main()
