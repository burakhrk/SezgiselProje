import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time  # Zaman kontrolü için time modülünü ekledik

# Veri yükleyici scriptini import eder
import tsp_data_loader
# Yardımcı fonksiyonlar scriptini (kaydetme, çizim vb.) import eder
import tsp_utils

# Şehir koordinatlarından mesafe matrisi oluşturur (Bu fonksiyon artık data_loader'da)
# def create_distance_matrix(DATA):
#     coords = DATA[:, 1:3] # Şehir koordinatlarını alır (genellikle 1. ve 2. sütunlar)
#     # Tüm şehir çiftleri arasındaki farkı hesaplar
#     diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
#     # Öklid mesafesini hesaplar
#     distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))
#     return distance_matrix

# Bir rotanın toplam mesafesini (fitness değerini) hesaplar
def calculate_tsp_fitness(permuted_array, DIST):
    # Rotadaki ardışık şehirler arasındaki mesafeleri toplar
    total_distance = DIST[permuted_array[:-1], permuted_array[1:]].sum()
    # Son şehirden ilk şehire dönüş mesafesini ekler (tur tamamlanır)
    total_distance += DIST[permuted_array[-1], permuted_array[0]]
    return total_distance

# Başlangıç için rastgele çözümlerden oluşan bir popülasyon oluşturur
def generate_random_solutions(N, PS, DIST):
    population = np.zeros((PS, N+1)) # Popülasyon matrisi (PS x N+1 boyutunda)
    for i in range(PS):
        perm = np.random.permutation(N) # Şehirlerin rastgele permütasyonunu oluşturur
        population[i, :N] = perm # Permütasyonu popülasyona ekler
        # Oluşturulan rotanın fitness değerini (toplam mesafe) hesaplar ve kaydeder
        population[i, N] = calculate_tsp_fitness(perm, DIST)
    return population

# Rulet Tekerleği Seçimi yöntemiyle popülasyondan iki ebeveyn seçer
def roulette_wheel_selection(F):
    F2 = 1 / (1 + F) # Fitness değerlerini tersine çevirir (daha düşük mesafe daha iyi fitness)
    pFitness = F2 / np.sum(F2) # Olasılık dağılımını hesaplar
    cFitness = np.cumsum(pFitness) # Kümülatif olasılık dağılımını hesaplar
    rn1, rn2 = np.random.rand(2) # İki rastgele sayı üretir
    # Rastgele sayılara göre ebeveyn indekslerini seçer
    b1 = np.searchsorted(cFitness, rn1)
    b2 = np.searchsorted(cFitness, rn2)
    return b1, b2 # Seçilen ebeveynlerin indekslerini döndürür

# İkili Turnuva Seçimi yöntemiyle popülasyondan iki ebeveyn seçer (şu anda kullanılmıyor)
def binary_tournament_selection(F):
    random_index1 = np.random.randint(len(F))
    random_index2 = np.random.randint(len(F))
    random_index3 = np.random.randint(len(F))
    random_index4 = np.random.randint(len(F))
    # İki rastgele bireyden daha iyi olanı seçer
    if F[random_index1] < F[random_index2]:
        b1 = random_index1
    else:
        b1 = random_index2
    # Başka iki rastgele bireyden daha iyi olanı seçer
    if F[random_index3] < F[random_index4]:
        b2 = random_index3
    else:
        b2 = random_index4
    return b1, b2 # Seçilen ebeveynlerin indekslerini döndürür

# Döngü Çaprazlama (Cycle Crossover - CX) uygular
def cycle_crossover(parent1, parent2):
    size = len(parent1) # Rota boyutu (şehir sayısı)
    offspring = -np.ones(size, dtype=int) # Yavru rotayı başlangıçta -1 ile doldurur
    start_index = np.random.randint(size) # Döngünün başlayacağı rastgele bir indeks seçer
    
    # Hangi ebeveynin döngüyü başlatacağına rastgele karar verir
    current_parent, other_parent = (parent1, parent2) if np.random.rand() > 0.5 else (parent2, parent1)
    
    val = current_parent[start_index] # Başlangıç noktasındaki değeri alır
    # Döngüyü takip eder ve şehirleri yavruya kopyalar
    while True:
        offspring[start_index] = current_parent[start_index]
        start_index = np.where(current_parent == other_parent[start_index])[0][0] # Karşı ebeveyndeki değeri bulur
        if current_parent[start_index] == val: # Döngü tamamlandıysa durur
            break

    # Döngüye dahil olmayan şehirleri diğer ebeveynden kopyalar
    offspring[offspring == -1] = other_parent[offspring == -1]

    return offspring # Oluşturulan yavru rotayı döndürür

# Sıra Çaprazlama (Order Crossover - OX) uygular
def order_crossover(parent1, parent2):
    size = len(parent1) # Rota boyutu
    offspring = np.empty(size, dtype=int) # Yavru rotayı oluşturur
    # Rastgele bir kesit noktası seçer
    start, end = sorted(np.random.choice(range(size), 2, replace=False))
    offspring[start:end+1] = parent1[start:end+1] # Seçilen kesiti ilk ebeveynden kopyalar
    
    p2_index = 0
    # Diğer ebeveyndeki şehirleri kesitte olmayan sırayla yavruya ekler
    for i in range(size):
        if i >= start and i <= end:
            continue # Kesit içindeyse atlar
        # Kesitte zaten olan şehri diğer ebeveynde atlar
        while parent2[p2_index] in offspring[start:end+1]:
            p2_index += 1
        offspring[i] = parent2[p2_index] # Şehri yavruya ekler
        p2_index += 1
    
    return offspring # Oluşturulan yavru rotayı döndürür

# Yer Değiştirme Mutasyonu (Swap Mutation) uygular
def swap_mutation(my_ofs):
    size = len(my_ofs) # Rota boyutu
    # Rastgele iki pozisyon seçer
    first_index, second_index = np.random.choice(size, 2, replace=False)
    # Seçilen pozisyonlardaki şehirleri yer değiştirir
    my_ofs[first_index], my_ofs[second_index] = my_ofs[second_index], my_ofs[first_index]
    return my_ofs # Mutasyona uğramış rotayı döndürür

# İki Opt Mutasyonu (Two-Opt Mutation) uygular
def two_opt_mutation(my_ofs):
    size = len(my_ofs) # Rota boyutu
    # Rastgele bir segment seçer
    start_index, end_index = sorted(np.random.choice(size, 2, replace=False))
    # Seçilen segmenti ters çevirir
    my_ofs[start_index:end_index+1] = my_ofs[start_index:end_index+1][::-1]
    return my_ofs # Mutasyona uğramış rotayı döndürür

# Ekleme Mutasyonu (Insert Mutation) uygular
def insert_mutation(my_ofs):
    size = len(my_ofs) # Rota boyutu
    # Rastgele bir şehri ve ekleneceği pozisyonu seçer
    from_index, to_index = np.random.choice(size, 2, replace=False)
    
    # Şehri eski pozisyonundan siler ve yeni pozisyona ekler
    # np.insert fonksiyonunun davranışı nedeniyle from_index ve to_index sırasına göre farklı işlem yapılır
    if from_index < to_index:
        my_ofs = np.insert(np.delete(my_ofs, from_index), to_index, my_ofs[from_index])
    else:
        my_ofs = np.insert(np.delete(my_ofs, from_index), to_index, my_ofs[from_index])
    
    return my_ofs # Mutasyona uğramış rotayı döndürür

# Şehirleri ve rotayı çizen fonksiyon
def plot_cities(cities, it, path, distance, fig, ax):
    ax.clear()
    ax.scatter(cities[:, 1], cities[:, 2], c='red', label='Cities', zorder=5)
    for i, (x, y) in enumerate(cities[:, 1:3]):
        ax.text(x, y, str(i), color="black", fontsize=9, zorder=5)
    if path is not None:
        path = np.array(path, dtype=int)
        path_coords = cities[path, 1:3]
        ax.plot(path_coords[:, 0], path_coords[:, 1], 'k-', lw=1, zorder=1)
        ax.plot([path_coords[-1, 0], path_coords[0, 0]], [path_coords[-1, 1], path_coords[0, 1]], 'k-', lw=1, label='Path', zorder=1)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Iteration: {it} Total Distance: {distance:.2f}')
    ax.legend()
    plt.draw()
    plt.pause(0.1)  # 0.1 saniye bekle

# Veri yükleme ve mesafe matrisi oluşturma kısmı kaldırıldı.
# file_name = os.path.join(os.path.dirname(__file__), 'datas', 'att48.txt')
# DATA_CITY = np.loadtxt(file_name, dtype=float)
# DIST_CITY = create_distance_matrix(DATA_CITY)

# Genetik algoritmanın ana fonksiyonu
# Fonksiyon artık N ve DIST_CITY'yi parametre olarak alıyor
def main(ER, CR, MR, N, DIST_CITY, DATA_CITY, seed=42, max_time=300):
    np.random.seed(seed)  # Rastgelelik için başlangıç tohumu (seed) ayarlar
    PS = N * 2  # Popülasyon boyutu (şehir sayısının 2 katı)
    ITERATION = 700  # Maksimum iterasyon sayısı
    
    # Başlangıç popülasyonunu oluşturur
    my_pop = generate_random_solutions(N, PS, DIST_CITY)
    it = 0  # İterasyon sayacı
    
    # Zaman kontrolü için başlangıç zamanını kaydet
    start_time = time.time()
    
    # Grafik çizimi için ayarlar
    plt.ion()  # İnteraktif modu aç
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ana genetik algoritma döngüsü
    while it < ITERATION:
        # Zaman kontrolü
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > max_time:
            print(f"\nZaman sınırına ulaşıldı: {elapsed_time:.2f} saniye")
            break
            
        it += 1
        my_pop = my_pop[np.argsort(my_pop[:, N])]

        # Güncel en iyi rotayı çizdirir
        plot_cities(DATA_CITY, it, my_pop[0, :N], my_pop[0, N], fig, ax)

        # Çaprazlama ile yeni yavrular oluşturur
        offspring_size = max(1, int(PS * CR))  # En az 1 yavru üret
        offspring = np.zeros((offspring_size, N + 1))  # Yavrular için boş matris
        
        # Belirlenen çaprazlama oranı (CR) kadar yavru üretir
        for i in range(offspring_size):
            b1, b2 = roulette_wheel_selection(my_pop[:, N])  # Rulet Tekerleği ile ebeveyn seçimi
            # Rastgele Cycle veya Order çaprazlama uygular
            if np.random.rand() < 0.5:
                my_ofs = cycle_crossover(my_pop[b1, :N], my_pop[b2, :N])
            else:
                my_ofs = order_crossover(my_pop[b1, :N], my_pop[b2, :N])
            
            # Belirlenen mutasyon oranı (MR) olasılığı ile mutasyon uygular
            if np.random.rand() < MR:
                mutselprob = np.random.rand()
                # Rastgele üç mutasyon türünden birini seçer ve uygular
                if mutselprob < 1/3:
                    my_ofs = swap_mutation(my_ofs)
                elif mutselprob < 2/3:
                    my_ofs = two_opt_mutation(my_ofs)
                else:
                    my_ofs = insert_mutation(my_ofs)
            
            # Oluşturulan yavruyu ve fitness değerini yavru matrisine ekler
            offspring[i, :N] = my_ofs
            offspring[i, N] = calculate_tsp_fitness(my_ofs, DIST_CITY)
        
        # Yeni popülasyonun elit ve yavru kısımlarının toplam boyutu
        elite_size = max(1, int(PS * ER))  # En az 1 elit birey
        my_pop_segment_size = elite_size + offspring_size

        # Popülasyon güncelleme / Yerine Koyma Stratejisi
        if my_pop_segment_size <= PS:  # Eğer toplam boyut popülasyon boyutundan küçük veya eşitse
            # Popülasyonun elit kısmı korunur
            new_pop = my_pop[:elite_size].copy()
            # Yavrular eklenir
            new_pop = np.vstack((new_pop, offspring))
            # Eksik kısım için rastgele çözümler eklenir
            if new_pop.shape[0] < PS:
                rand_sol_size = PS - new_pop.shape[0]
                rand_sol = generate_random_solutions(N, rand_sol_size, DIST_CITY)
                new_pop = np.vstack((new_pop, rand_sol))
            my_pop = new_pop
        else:  # Eğer toplam boyut popülasyon boyutundan büyükse
            # Elit bireyleri koru
            new_pop = my_pop[:elite_size].copy()
            # Yavruları ekle (popülasyon boyutuna göre kırp)
            remaining_size = PS - elite_size
            if remaining_size > 0:
                new_pop = np.vstack((new_pop, offspring[:remaining_size]))
            my_pop = new_pop

    # Döngü bittiğinde popülasyon en iyiye göre sıralanmıştır, en iyi fitness değerini döndürür
    final_time = time.time() - start_time
    print(f"Toplam çalışma süresi: {final_time:.2f} saniye")
    print(f"Toplam iterasyon sayısı: {it}")
    return my_pop[0, N], it  # Final mesafe ve iterasyon sayısını döndür

# Kodun ana çalıştırma bloğu
if __name__ == "__main__":
    # Veri dosyasının yolunu belirler (bu scriptin bulunduğu klasöre göre ayarlanır)
    data_file_name = os.path.join(os.path.dirname(__file__), 'datas', 'att48.txt')
    
    # Veri yükleyiciyi kullanarak şehir sayısını ve mesafe matrisini yükler
    N_cities, dist_matrix = tsp_data_loader.load_tsp_data(data_file_name)
    
    # Eğer veri başarıyla yüklendiyse algoritmayı çalıştırır
    if N_cities is not None and dist_matrix is not None:
        # Denenecek ER (Elitizm Oranı), CR (Çaprazlama Oranı), MR (Mutasyon Oranı) değerleri
        ER_values = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
        CR_values = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
        MR_values = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

        results = [] # Sonuçları saklamak için boş liste
        # Tüm ER, CR, MR kombinasyonlarını döngüyle dener
        for ER in ER_values:
            for CR in CR_values:
                for MR in MR_values:
                    try:
                        # Eğer PS * CR < 1 ise yavru üretilmez, bu durumu atlar
                        # PS burada N'e eşit olduğu için N * CR kontrolü yapılır
                        if int(N_cities * CR) < 1 and CR > 0:
                             print(f"Skipping ER: {ER}, CR: {CR}, MR: {MR}. N * CR is less than 1, no offspring will be generated.")
                             continue

                        # Genetik algoritmayı belirtilen parametrelerle ve yüklenen veri ile çalıştırır
                        result = main(ER, CR, MR, N_cities, dist_matrix, dist_matrix)
                        # Sonucu listeye ekler
                        results.append((ER, CR, MR, result))
                        # Güncel kombinasyonun sonucunu yazdırır
                        print(f"ER: {ER}, CR: {CR}, MR: {MR}, Final Distance: {result}")
                    except Exception as e: # Hata oluşursa atlar ve hata mesajını yazdırır
                        print(f"Skipping ER: {ER}, CR: {CR}, MR: {MR} due to error: {e}")
        
        # Sonuçları yardımcı fonksiyonu kullanarak kaydeder
        output_excel_file = 'att48_results.xlsx'
        tsp_utils.save_results_to_excel(results, output_excel_file)

    else:
        print("Veri yüklenemediği için algoritma çalıştırılamadı.") 