import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from proses import cleaning, case_folding, normalisasi_singkatan,tokenization,stemming,remove_stopwords,tfidf,smoteMiSVM,smoteSVM,smote

def preprocess_data(data):
    
    st.subheader("Data setelah di preprocessing")

    st.subheader("Cleaning")
    data["teks"] = data["teks"].apply(cleaning)
    st.write(data)

    st.subheader("Case Folding")
    data["teks"] = data["teks"].apply(case_folding)
    st.write(data)

    st.subheader("Normalization")
    data["teks"] = data["teks"].apply(normalisasi_singkatan)
    st.write(data)

    st.subheader("Tokenization")
    data["teks"] = data["teks"].apply(tokenization)
    st.write(data)

    st.subheader("Stopword Removal")
    data["teks"] = data["teks"].apply(remove_stopwords)
    st.write(data)

    st.subheader("Stemming")
    data["teks"] = data["teks"].apply(stemming)
    st.write(data)

    return data

def split_and_tfidf(data):
    data_tfidf = tfidf(data)

    X_train, X_test, y_train, y_test = train_test_split(
        data_tfidf, data["Label"], test_size=0.3, random_state=42, stratify=data["Label"]
    
   )
 
    return X_train, X_test, y_train, y_test

def svm_results(metode,c, X_train, y_train, X_test, y_test, data):
    # SVM + SMOTE + MI
    
    if metode == "SMOTE + MI + SVM":
        st.subheader("SVM With SMOTE & MI")

        st.subheader("Oversampling")
        
        st.text(f"Before Oversampling, Label 2: {sum(y_train == 2)}")
        st.text(f"Before Oversampling, Label 1: {sum(y_train == 1)}")
        st.text(f"Before Oversampling, Label 0: {sum(y_train == 0)}")
        
        X_train_res, y_train_res = smote(X_train, y_train)
        
        st.text(f"After Oversampling, Label 2: {sum(y_train_res == 2)}")
        st.text(f"After Oversampling, Label 1: {sum(y_train_res == 1)}")
        st.text(f"After Oversampling, Label 0: {sum(y_train_res == 0)}")
        
        predicted = smoteMiSVM(c, X_train_res, y_train_res, X_test,y_test)
 
        st.text("SVM Training Metrics:")
        train_predicted = smoteMiSVM(c, X_train_res, y_train_res, X_train, y_train)
        st.text(f"SVM Accuracy: {accuracy_score(y_train, train_predicted)}")
        st.text(
            f"SVM Precision: {precision_score(y_train, train_predicted, average='weighted')}"
        )
        st.text(
            f"SVM Recall: {recall_score(y_train, train_predicted, average='weighted')}"
        )
        st.text(f"SVM F1 Score: {f1_score(y_train, train_predicted, average='weighted')}")
        st.text("Confusion Matrix:")
        st.write(confusion_matrix(y_train, train_predicted))
    
        report = classification_report(y_train, train_predicted)
        st.write("Classification Report")
        st.code(report,language='markdown')

        st.text("SVM Testing Metrics:")
        st.text(f"SVM Accuracy: {accuracy_score(y_test, predicted)}")
        st.text(
            f"SVM Precision: {precision_score(y_test, predicted, average='weighted')}"
        )
        st.text(
            f"SVM Recall: {recall_score(y_test, predicted, average='weighted')}"
        )
        st.text(f"SVM F1 Score: {f1_score(y_test, predicted, average='weighted')}")
        st.text("Confusion Matrix:")
        st.write(confusion_matrix(y_test, predicted))
        
        report = classification_report(y_test, predicted)
        st.write("Classification Report")
        st.code(report,language='markdown')

        st.subheader("Classified Labels and Headlines")
        selected_rows = X_test[predicted == 0].nonzero()[0]
        selected_rows2 = X_test[predicted == 1].nonzero()[0]
        selected_rows3 = X_test[predicted == 2].nonzero()[0]

        # definisikan mapping label angka ke teks
        label_map = {0: 'normal', 1: 'penipuan', 2: 'promo'} 
        
        selected_rows = np.concatenate([selected_rows, selected_rows2, selected_rows3]) 
        predicted_series = pd.Series(predicted[selected_rows])

        # gunakan mapping untuk label teks
        result_df = pd.DataFrame({
            "Classified Label": predicted_series.map(label_map),  
            "teks": data.loc[selected_rows, "teks"].values
        })

        result_df = result_df.drop_duplicates(subset=['teks'])
        st.write(result_df)

    # SVM Dengan SMOTE
    elif metode == "SMOTE + SVM":
        st.subheader("SVM With SMOTE")

        st.subheader("Oversampling")
        
        st.text(f"Before Oversampling, Label 2: {sum(y_train == 2)}")
        st.text(f"Before Oversampling, Label 1: {sum(y_train == 1)}")
        st.text(f"Before Oversampling, Label 0: {sum(y_train == 0)}")
        
        X_train_res, y_train_res = smote(X_train, y_train)
        predicted = smoteSVM(c, X_train_res, y_train_res, X_test,y_test)

        st.text(f"After Oversampling, Label 2: {sum(y_train_res == 2)}")
        st.text(f"After Oversampling, Label 1: {sum(y_train_res == 1)}")
        st.text(f"After Oversampling, Label 0: {sum(y_train_res == 0)}")

        st.text("SVM Training Metrics:")
        train_predicted = smoteSVM(c, X_train_res, y_train_res, X_train,y_train)
        st.text(f"SVM Accuracy: {accuracy_score(y_train, train_predicted)}")
        st.text(
            f"SVM Precision: {precision_score(y_train, train_predicted, average='weighted')}"
        )
        st.text(
            f"SVM Recall: {recall_score(y_train, train_predicted, average='weighted')}"
        )
        st.text(f"SVM F1 Score: {f1_score(y_train, train_predicted, average='weighted')}")
        st.text("Confusion Matrix:")
        st.write(confusion_matrix(y_train, train_predicted))

        report = classification_report(y_train, train_predicted)
        st.write("Classification Report")
        st.code(report,language='markdown')

        st.text("SVM Testing Metrics:")
        st.text(f"SVM Accuracy: {accuracy_score(y_test, predicted)}")
        st.text(
            f"SVM Precision: {precision_score(y_test, predicted, average='weighted')}"
        )
        st.text(
            f"SVM Recall: {recall_score(y_test, predicted, average='weighted')}"
        )
        st.text(f"SVM F1 Score: {f1_score(y_test, predicted, average='weighted')}")
        st.text("Confusion Matrix:")
        st.write(confusion_matrix(y_test, predicted))

        report = classification_report(y_test, predicted)
        st.write("Classification Report")
        st.code(report,language='markdown')
        
        st.subheader("Classified Labels and Headlines")
        selected_rows = X_test[predicted == 0].nonzero()[0]
        selected_rows2 = X_test[predicted == 1].nonzero()[0]
        selected_rows3 = X_test[predicted == 2].nonzero()[0]

        # definisikan mapping label angka ke teks
        label_map = {0: 'normal', 1: 'penipuan', 2: 'promo'} 
        
        selected_rows = np.concatenate([selected_rows, selected_rows2, selected_rows3]) 
        predicted_series = pd.Series(predicted[selected_rows])

        # gunakan mapping untuk label teks
        result_df = pd.DataFrame({
            "Classified Label": predicted_series.map(label_map),  
            "teks": data.loc[selected_rows, "teks"].values
            
        })

        result_df = result_df.drop_duplicates(subset=['teks'])
        st.write(result_df)

