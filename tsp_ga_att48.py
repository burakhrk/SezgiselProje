import os
import numpy as np
import pandas as pd
import tsp_data_loader
import tsp_utils
from tsp_ga_solver import main

# Deney parametreleri
EXPERIMENT_NUMBER = 1
RANDOM_SEED = 42
MAX_TIME = 100  # Maksimum çalışma süresi (saniye)

# Şehir verilerini yükle
file_path = os.path.join("Datas", "att48.txt")
N_cities, distance_matrix = tsp_data_loader.load_tsp_data(file_path)
if N_cities is None or distance_matrix is None:
    print("Veri yüklenemedi. Program sonlandırılıyor.")
    exit(1)

# Deney parametreleri - optimize edilmiş değerler
elite_rates = [0.3, 0.35, 0.4]      # Elitizm oranı
crossover_rates = [0.4, 0.45, 0.5]  # Çaprazlama oranı
mutation_rates = [0.5, 0.55, 0.6]  # Mutasyon oranı
population_multiplier = 10  # Popülasyon çarpanı (N * multiplier)

# Sonuçları saklamak için liste
results = []

# Tek kombinasyonu test et
for er in elite_rates:
    for cr in crossover_rates:
        for mr in mutation_rates:
            print(f"\nDeney {EXPERIMENT_NUMBER} - Parametreler:")
            print(f"ER: {er}, CR: {cr}, MR: {mr}")
            print(f"Popülasyon Boyutu: {N_cities * population_multiplier}")
            
            # Genetik algoritmayı çalıştır
            final_distance, iteration_count = main(
                ER=er,
                CR=cr,
                MR=mr,
                N=N_cities,
                DIST_CITY=distance_matrix,
                DATA_CITY=distance_matrix,  # Mesafe matrisini kullan
                seed=RANDOM_SEED,
                max_time=MAX_TIME,
                population_multiplier=population_multiplier
            )
            
            # Sonuçları kaydet
            results.append({
                'Deney': EXPERIMENT_NUMBER,
                'ER': er,
                'CR': cr,
                'MR': mr,
                'Popülasyon': N_cities * population_multiplier,
                'İterasyon': iteration_count,
                'Mesafe': final_distance
            })

# Sonuçları DataFrame'e dönüştür
results_df = pd.DataFrame(results)
output_file = f'experiment_{EXPERIMENT_NUMBER}_seed_{RANDOM_SEED}_time_{MAX_TIME}.xlsx'

# Eğer dosya varsa, mevcut verileri oku ve yeni verileri ekle
if os.path.exists(output_file):
    existing_df = pd.read_excel(output_file)
    combined_df = pd.concat([existing_df, results_df], ignore_index=True)
    combined_df.to_excel(output_file, index=False)
else:
    # Dosya yoksa yeni dosya oluştur
    results_df.to_excel(output_file, index=False)

print(f"\nSonuçlar {output_file} dosyasına eklendi.") 