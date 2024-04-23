import streamlit as st
import pandas as pd 
from klasifikasi import preprocess_data,split_and_tfidf,svm_results
st.title("""KLASIFIKASI SMS """)

def upload_dataset():
    upload_file = st.file_uploader("Upload Dataset dalam bentuk CSV", type=["csv"])
    if upload_file is not None:
        dataset = pd.read_csv(upload_file, sep=';', encoding='ISO-8859-1')
        return dataset
    else:
        st.warning("Dataset Not Found")
    return None 

st.subheader("Input Dataset")
data = upload_dataset()
    
# nilai C
pilihan_c = [0.1, 0.5, 1, 5, 10]
c = st.selectbox("Nilai C", pilihan_c)

metode_input = st.selectbox("Pilih Metode", [" ", "SMOTE + MI + SVM", "SMOTE + SVM"])

klasifikasi_button = st.button("Klasifikasi")

if klasifikasi_button and c and metode_input != "":
    # Preprocessing
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_and_tfidf(data)
    
    # Proses klasifikasi
    svm_results(metode_input, c, X_train, y_train, X_test, y_test, data)

