import os
import numpy as np
import pandas as pd
import tsp_data_loader
import tsp_utils
from tsp_ga_solver import main

# Deney parametreleri
EXPERIMENT_NUMBER = 1
RANDOM_SEED = 42
MAX_TIME = 200  # Maksimum çalışma süresi (saniye)

# Şehir verilerini yükle
file_path = os.path.join("Datas", "att48.txt")
cities = tsp_data_loader.load_tsp_data(file_path)
if cities is None:
    print("Veri yüklenemedi. Program sonlandırılıyor.")
    exit(1)

# Mesafe matrisini oluştur
distance_matrix = tsp_data_loader.create_distance_matrix(cities)

# Deney parametreleri - optimize edilmiş değerler
elite_rates = [0.3]      # Elitizm oranı
crossover_rates = [0.5]  # Çaprazlama oranı
mutation_rates = [0.25]  # Mutasyon oranı
population_multiplier = 10  # Popülasyon çarpanı (N * multiplier)

# Sonuçları saklamak için liste
results = []

# Tek kombinasyonu test et
for er in elite_rates:
    for cr in crossover_rates:
        for mr in mutation_rates:
            print(f"\nDeney {EXPERIMENT_NUMBER} - Parametreler:")
            print(f"ER: {er}, CR: {cr}, MR: {mr}")
            print(f"Popülasyon Boyutu: {len(cities) * population_multiplier}")
            
            # Genetik algoritmayı çalıştır
            final_distance, iteration_count = main(
                ER=er,
                CR=cr,
                MR=mr,
                N=len(cities),
                DIST_CITY=distance_matrix,
                DATA_CITY=cities,
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
                'Popülasyon': len(cities) * population_multiplier,
                'İterasyon': iteration_count,
                'Mesafe': final_distance
            })

# Sonuçları DataFrame'e dönüştür ve Excel'e kaydet
results_df = pd.DataFrame(results)
output_file = f'experiment_{EXPERIMENT_NUMBER}_seed_{RANDOM_SEED}_time_{MAX_TIME}.xlsx'
results_df.to_excel(output_file, index=False)

print(f"\nSonuçlar {output_file} dosyasına kaydedildi.") 