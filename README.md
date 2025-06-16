# Forecasting Nilai CIF Menggunakan LSTM

Proyek ini bertujuan untuk melakukan prediksi nilai Cost, Insurance, and Freight (CIF) dalam USD berdasarkan data historis bulanan menggunakan metode Long Short-Term Memory (LSTM), sebuah algoritma deep learning yang efektif dalam memproses data deret waktu (time series).

---

## 📂 Dataset

- Dataset: `data.csv`
- Format: CSV dengan delimiter `;`
- Kolom yang digunakan: `Waktu` (periode bulan dan tahun) dan `Nilai CIF (US$)`
- Data diolah menjadi `PeriodIndex` berdasarkan `Waktu`.

---

## 🧹 Pra-pemrosesan Data

1. Menghapus kolom tak berguna (`Unnamed`).
2. Mengonversi kolom `Waktu` ke format bulanan (`PeriodIndex`).
3. Visualisasi nilai CIF dari waktu ke waktu menggunakan `matplotlib` dan `seaborn`.
4. Uji stasioneritas menggunakan **Augmented Dickey-Fuller (ADF Test)**.

---

## ⚙️ Proses Modelling

### 1. Data Splitting
- 80% data sebagai **training set**
- 20% data sebagai **testing set**

### 2. Normalisasi
- Menggunakan `MinMaxScaler` untuk menyesuaikan nilai CIF ke skala [0, 1].

### 3. Format Data Supervised
- Menggunakan `time_step=60` untuk membuat data sequence.
- Bentuk data menjadi format `[samples, time_steps, features]` agar sesuai dengan input LSTM.

### 4. Arsitektur Model LSTM
```python
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
```

### 5. Training
- Optimizer: `Adam`
- Loss Function: `mean_squared_error`
- Epochs: 50 (atau 100 dengan EarlyStopping)
- Batch Size: 32
- Validasi dengan data `validation_split`

---

## 🔮 Future Forecasting

- Memprediksi **12 bulan ke depan** dari data terakhir
- Visualisasi prediksi dengan grafik garis berwarna merah

---

## 🧪 Evaluasi Model

Metrik evaluasi pada data test:

- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)

---

## 📊 Visualisasi

- Grafik nilai CIF aktual dan prediksi masa depan
- Visualisasi tren CIF historis menggunakan line plot

---
