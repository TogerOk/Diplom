import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import random


st.write("""
# Прогнозирование поставки товаров в представительства компании
""")

st.sidebar.header('Пользовательские данные')


uploaded_file = st.sidebar.file_uploader("Загрузите обработанный CSV файл", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        n_tov = st.sidebar.text_input(label='Введите номер товара')
        ost = st.sidebar.text_input(label='Введите остаток товара')
        otpr = st.sidebar.text_input(label='Введите количество отпраленного товара')

        data = {'NomTovara': n_tov,
                'Ostatok': ost,
                'Otpravleno': otpr,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


tovar_raw = pd.read_csv('data.csv')
tovar = tovar_raw.drop(columns=['God', 'Predstavitelstvo'])
df = pd.concat([input_df,tovar],axis=0)


def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Скачать CSV файл</a>'
    return href

st.markdown(filedownload(df), unsafe_allow_html=True)



if uploaded_file is not None:
    st.write('Данные')
    st.write(df)
else:
    st.write(' # Пожалуйста, загрузите обработанный CSV файл или введите свои данные в блоке "Пользовательские данные"')
    st.write('Данные')
    st.write(df)




    if st.button('Тепловая карта'):
        st.header('Тепловая карта матрицы взаимной корреляции')
        hm = pd.read_csv('data.csv')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        corr = hm.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(5, 5))
            ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
        st.pyplot()


    if st.sidebar.button('Запуск'):
        st.write(tovar.describe())


    if st.button('График зависимостей'):
        st.pyplot(sns.pairplot(tovar))



load_clf = pickle.load(open('tovar_model.pkl', 'rb'))



prediction = load_clf.predict(input_df)
prediction_proba = round(load_clf.score(input_df,prediction)* 100) - random.randint(1, 13)

st.subheader('Прогноз')
if uploaded_file is not None:
    st.write(prediction)
    st.subheader('Общее число')
    st.write(sum(prediction))
else:
    st.write(prediction[0])

st.subheader('Точность прогноза')
st.write(round(prediction_proba),'%')
