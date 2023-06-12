import streamlit as st
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

#navigasi sidebar
# horizontal menu
selected2 = option_menu(None, ["Data", "Procecing data", "Modelling", 'Implementasi'], 
    icons=['house', 'cloud-upload', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
selected2

#halaman Data
if (selected2 == 'Data') :
    st.title('deskripsi data')

    st.write("Ini adalah contoh data yang tersedia dalam aplikasi Streamlit.")
    st.write("Data ini berisi informasi tentang sebuah bank (Bank Thera) yang manajemennya ingin mencari cara untuk mengubah pelanggan pertanggungjawabannya menjadi pelanggan pinjaman pribadi (sambil mempertahankan mereka sebagai deposan")
    st.write("Data ini diambil dari kaggle")
    st.write("Data ini merupakan type data Numerik")
    data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
    st.write(data)



#halaman procecing data 
if (selected2 == 'Procecing data') :
    st.title('Procecing Data')

    st.write("saya menggunakan procecing data SKALASI STANDAR ")
    st.write("Dengan hasil procecing data")
    data = pd.read_csv('preprocessed_data.csv')
    st.write(data)

#halaman modelling
if (selected2 == 'Modelling'):
    st.title('Modelling')

    pilih = st.radio('Pilih', ('Naive Bayes', 'Decision Tree', 'KNN', 'ANN'))

    if (pilih == 'Naive Bayes'):
        st.title(' Nilai Akurasi 70,8%')
    elif (pilih == 'Decision Tree'):
        st.title(' Nilai Akurasi 75%')
    elif (pilih == 'KNN'):
        st.title(' Nilai Akurasi 65%')
    elif (pilih == 'ANN'):
        st.title(' Nilai Akurasi 73,8%')


#halaman implementasi
# Load the saved model
if (selected2 == 'Implementasi'):
    st.title('Implementasi')


    model_filename = 'loan.pkl'
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)

    # Create a function to preprocess the input data
    def preprocess_input(data):
        # Convert the input data into a DataFrame
        df = pd.DataFrame(data, index=[0])
        
        # Standarisasi fitur
        scaler = StandardScaler()
        scaler.fit(df)  # Fit the scaler on the input data
        df_std = scaler.transform(df)
        
        return df_std

    # Create the Streamlit web app
    def main():
        st.title('Personal Loan Bank')

        # Create input fields for the features
        id = st.text_input('ID')
        age = st.number_input('Umur', min_value=0, max_value=100, value=30)
        experience = st.number_input('Pengalaman', min_value=0, max_value=100, value=10)
        income = st.number_input('Pendapatan', min_value=0, value=50000)
        zip_code = st.number_input('Kode Pos', min_value=0, value=10000)
        family = st.number_input('Keluarga', min_value=0, value=1)
        cc_avg = st.number_input('CCAvg', min_value=0, value=1)
        education = st.number_input('Pendidikan', min_value=1, max_value=3, value=1)
        mortgage = st.number_input('Mortgage', min_value=0, value=0)
        personal_loan = st.number_input('Pinjaman Personal', min_value=0, max_value=1, value=0)
        securities_account = st.selectbox('Rekening Efek', [0, 1])
        cd_account = st.selectbox('Rekening CD', [0, 1])
        online = st.selectbox('Online', [0, 1])


        # Create a dictionary with the input data
        input_data = {
            'ID': id,
            'Age': age,
            'Experience': experience,
            'Income': income,
            'ZIP Code': zip_code,
            'Family': family,
            'CCAvg': cc_avg,
            'Education': education,
            'Mortgage': mortgage,
            'Personal Loan': personal_loan,
            'Securities Account': securities_account,
            'CD Account': cd_account,
            'Online': online,
            
        }

        # Perform prediction when the button is pressed
        if st.button('Hitung'):
            # Preprocess the input data
            input_data_std = preprocess_input(input_data)
            
            # Predict using the loaded model
            prediction = model.predict(input_data_std)

            # Display the prediction
            if prediction[0] == 0:
                st.error('Tidak Memenuhi Syarat untuk Pinjaman Personal')
            else:
                st.success('Memenuhi Syarat untuk Pinjaman Personal')

    if __name__ == '__main__':
        main()
