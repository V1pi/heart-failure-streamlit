import pandas as pd
import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Função para preprocessamento, mesmo preprocessamento para todos os dados, incluindo os novos
def preprocessamento(df, sc, fit=False):
    # Dei uma olhada no google sobre esses dado medicos, se tiverem normais eu coloco 1, 
    # se não tiverem, eu coloco 0
    df.loc[:,'platelets'] = df.apply (lambda row: 0 if row['platelets'] >= 150000 and row['platelets'] <= 450000 else 1, axis=1)
    df.loc[:,'serum_sodium'] = df.apply (lambda row: 0 if row['serum_sodium'] >= 135 and row['serum_sodium'] <= 147 else 1, axis=1)
    # Para os outros dados eu faço o standard scaler
    standard_scaler_features = ['creatinine_phosphokinase','ejection_fraction', 'serum_creatinine']
    # O fit e feito apenas na hora de treinar o modelo
    if fit is True:
        sc.fit(df[standard_scaler_features])
    # Atualizo o dataframe com os dados preprocessados
    df.loc[:,standard_scaler_features] = sc.transform(df[standard_scaler_features])
    features=['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
    # Seleciona so as features
    x=df[features]
    return x

# Ler os dados do usuario
def user_input_features(xCol, sc):
    data = {}
    # Os dados que sao radio buttons sao os abaixos, os outros sao slider
    radioButtons = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
    for col in xCol.columns.to_list():
        if col in radioButtons:
            # Caso seja sexo mostra uns campos diferentes, caso contrario um "sim" ou "nao"
            if(col == 'sex'):
                r = st.sidebar.radio('sex', ['male', 'female'])
                data[col] = 1.0 if r == 'male' else 0
            else:
                r = st.sidebar.radio(col, ['Yes', 'No'])
                data[col] = 1.0 if r == 'Yes' else 0
        else:
            xCol[col] = xCol[col].astype(float, errors = 'raise')
            # a idade e tratada como inteiro, enquanto os outros sao floats
            if(col == 'age'):
                data[col] = st.sidebar.slider(col,int(xCol[col].min()),int(xCol[col].max()),int(xCol[col].mean()))
            else:  
                data[col] = st.sidebar.slider(col,xCol[col].min(),xCol[col].max(),float(xCol[col].mean()))  
    # Transformo os dados em um dataframe
    return pd.DataFrame(data, index=[0])

# Função para carregar os dados
def load_data():
    # Carrega os dados do dataset
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    # Separa as colunas que sao features e as que sao labels
    features=['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
    x=df[features]
    y=df['DEATH_EVENT']
    return x,y

# Função para treinar o modelo
def model_ia(x, y):
    # Cria o modelo
    model = KNeighborsClassifier(n_neighbors=4)
    # Treina o modelo
    model.fit(x,y)
    return model

# Função principal
def main():
    # Carrega os dados
    x_without_pre_process,y = load_data()
    # Cria o scaler
    sc = StandardScaler()
    # Preprocessa os dados
    x = preprocessamento(x_without_pre_process.copy(), sc, fit= True)

    # Printa a mensagem na pagina
    st.header('Predição de morte por infarto do coração')
    st.write("""
    Para informações sobre o dataset, acesse: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data""")

    # Carrega o modelo
    model = model_ia(x,y)
    
    # Mostra o texto na pagina
    st.sidebar.header('Escolha de paramentros para Predição')
    # Pega os inputs do usuario
    df = user_input_features(x_without_pre_process, sc)

    # Preprocesso os dados
    df_processado = preprocessamento(df.copy(), sc)

    # Mostra os parametros selecionados do usuario
    st.header('Parametros especificados')
    st.write("Antes do pré-processamento:")
    st.write(df)
    st.write("Depois do pré-processamento:")
    st.write(df_processado)

    # Faz a predição
    prediction = model.predict(df_processado)

    # Mostra a predição
    st.header('Risco de infarto')
    st.write("Resultado: {}".format("Sim" if prediction == 1 else "Não"))
    st.write('---')

if __name__=='__main__':
    main()