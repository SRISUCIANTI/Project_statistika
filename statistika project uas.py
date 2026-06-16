import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ==========================================
# 1. LOAD DATASET
# ==========================================

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

file_path = os.path.join(current_dir, "BIKE DETAILS.csv")

if not os.path.exists(file_path):
    print(f"[ERROR] File tidak ditemukan di: {file_path}")
    print("Pastikan file 'BIKE DETAILS.csv' berada dalam folder project.")
    exit()

df = pd.read_csv(file_path)

print("=== INFORMASI DATASET ===")
print(df.info())

print("\n=== 5 DATA PERTAMA ===")
print(df.head())

# ==========================================
# 2. PREPROCESSING
# ==========================================

X = df[['year']]
y = df['selling_price']

X = X.fillna(X.mean())
y = y.fillna(y.mean())

# ==========================================
# 3. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# 4. TRAINING MODEL
# ==========================================

model = LinearRegression()
model.fit(X_train, y_train)

# ==========================================
# 5. PREDIKSI
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 6. EVALUASI MODEL
# ==========================================

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== HASIL EVALUASI MODEL ===")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# ==========================================
# 7. ANALISIS REGRESI
# ==========================================

intercept = model.intercept_
koefisien = model.coef_[0]

print("\n=== PERSAMAAN REGRESI ===")
print(
    f"Harga_Jual = {intercept:.2f} + "
    f"({koefisien:.2f} × Tahun_Keluaran)"
)

print("\n=== INTERPRETASI ===")
print(
    f"Setiap kenaikan 1 tahun keluaran motor "
    f"diperkirakan mengubah harga jual sebesar "
    f"{koefisien:.2f} rupiah."
)

# ==========================================
# 8. VISUALISASI
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    X,
    y,
    alpha=0.7,
    label="Data Aktual"
)

plt.plot(
    X,
    model.predict(X),
    color="red",
    linewidth=2,
    label="Garis Regresi"
)

plt.title("Regresi Linear Sederhana Harga Motor Bekas")
plt.xlabel("Tahun Keluaran")
plt.ylabel("Harga Jual Motor")
plt.legend()
plt.grid(True)

# ==========================================
# 9. MEMBUAT FOLDER OUTPUT
# ==========================================

output_folder = os.path.join(current_dir, "output_analisis")
os.makedirs(output_folder, exist_ok=True)

# ==========================================
# 10. MENYIMPAN GRAFIK
# ==========================================

plt.savefig(
    os.path.join(
        output_folder,
        "grafik_regresi_motor_bekas.png"
    ),
    dpi=300
)

print("\n[INFO] Grafik berhasil disimpan.")

plt.show()

# ==========================================
# 11. MENYIMPAN METRIK EVALUASI
# ==========================================

with open(
    os.path.join(
        output_folder,
        "hasil_evaluasi_model.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write("=================================\n")
    f.write("HASIL EVALUASI REGRESI LINEAR\n")
    f.write("=================================\n")
    f.write(f"MAE  : {mae:.2f}\n")
    f.write(f"MSE  : {mse:.2f}\n")
    f.write(f"RMSE : {rmse:.2f}\n")
    f.write(f"R²   : {r2:.4f}\n")

print("[INFO] Metrik evaluasi berhasil disimpan.")

# ==========================================
# 12. MENYIMPAN PERSAMAAN REGRESI
# ==========================================

with open(
    os.path.join(
        output_folder,
        "persamaan_regresi.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write("PERSAMAAN REGRESI LINEAR\n")
    f.write("========================\n")
    f.write(
        f"Harga_Jual = {intercept:.2f} + "
        f"({koefisien:.2f} × Tahun_Keluaran)"
    )

print("[INFO] Persamaan regresi berhasil disimpan.")

# ==========================================
# 13. MENYIMPAN AKTUAL VS PREDIKSI
# ==========================================

df_hasil = pd.DataFrame({
    'Harga_Aktual': y_test.values,
    'Harga_Prediksi': y_pred
})

df_hasil.to_csv(
    os.path.join(
        output_folder,
        "perbandingan_aktual_vs_prediksi.csv"
    ),
    index=False
)

print("[INFO] Data aktual vs prediksi berhasil disimpan.")

# ==========================================
# 14. MENYIMPAN KOEFISIEN REGRESI
# ==========================================

df_koef = pd.DataFrame({
    'Parameter': ['Intercept', 'Koefisien Tahun'],
    'Nilai': [intercept, koefisien]
})

df_koef.to_csv(
    os.path.join(
        output_folder,
        "koefisien_regresi.csv"
    ),
    index=False
)

print("[INFO] Koefisien regresi berhasil disimpan.")

# ==========================================
# 15. MENYIMPAN LAPORAN ANALISIS
# ==========================================

with open(
    os.path.join(
        output_folder,
        "laporan_analisis.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write("ANALISIS REGRESI LINEAR SEDERHANA\n")
    f.write("=============================================\n\n")

    f.write("Judul Penelitian:\n")
    f.write("Prediksi Harga Jual Motor Bekas Berdasarkan Tahun Keluaran\n\n")

    f.write(f"Jumlah Data : {len(df)}\n\n")

    f.write("Persamaan Regresi:\n")
    f.write(
        f"Harga_Jual = {intercept:.2f} + "
        f"({koefisien:.2f} × Tahun_Keluaran)\n\n"
    )

    f.write("Evaluasi Model:\n")
    f.write(f"MAE  : {mae:.2f}\n")
    f.write(f"MSE  : {mse:.2f}\n")
    f.write(f"RMSE : {rmse:.2f}\n")
    f.write(f"R²   : {r2:.4f}\n\n")

    f.write("Interpretasi:\n")
    f.write(
        f"Setiap kenaikan satu tahun keluaran motor "
        f"diperkirakan mengubah harga jual sebesar "
        f"{koefisien:.2f} rupiah.\n"
    )

print("[INFO] Laporan analisis berhasil disimpan.")

# ==========================================
# 16. PREDIKSI DATA BARU
# ==========================================

tahun = int(input("\nMasukkan Tahun Keluaran Motor: "))

data_baru = pd.DataFrame(
    [[tahun]],
    columns=['year']
)

prediksi = model.predict(data_baru)

print(
    f"\nPrediksi Harga Jual Motor Tahun {tahun}"
)

print(
    f"Rp {prediksi[0]:,.0f}"
)

# ==========================================
# 17. MENYIMPAN HASIL PREDIKSI USER
# ==========================================

with open(
    os.path.join(
        output_folder,
        "hasil_prediksi_user.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write("HASIL PREDIKSI MOTOR BEKAS\n")
    f.write("==========================\n")
    f.write(f"Tahun Keluaran : {tahun}\n")
    f.write(f"Prediksi Harga : Rp {prediksi[0]:,.0f}\n")

print("[INFO] Hasil prediksi user berhasil disimpan.")

print("\n======================================")
print("SEMUA HASIL BERHASIL DISIMPAN")
print(f"Lokasi: {output_folder}")
print("======================================")