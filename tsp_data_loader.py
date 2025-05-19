import numpy as np
import os

# Şehir koordinatlarından mesafe matrisi oluşturur
def create_distance_matrix(DATA):
    # DATA'nın numpy array olduğundan emin olalım
    if isinstance(DATA, tuple):
        DATA = DATA[0]  # Tuple'dan ilk elemanı al (şehir koordinatları)
    
    coords = DATA[:, 1:3]  # Şehir koordinatlarını alır (genellikle 1. ve 2. sütunlar)
    # Tüm şehir çiftleri arasındaki farkı hesaplar
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    # Öklid mesafesini hesaplar
    distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    return distance_matrix

# Belirtilen dosya yolundaki TSP verisini yükler ve mesafe matrisini döndürür
def load_tsp_data(file_path):
    try:
        # Veriyi numpy array olarak yükle
        data = np.loadtxt(file_path, dtype=float)
        print(f"{file_path} verisi başarıyla yüklendi. Şehir sayısı: {len(data)}")
        return data
    except Exception as e:
        print(f"Hata: {file_path} dosyası yüklenirken bir sorun oluştu: {e}")
        return None

# Bu script doğrudan çalıştırıldığında örnek olarak att48 verisini yükler
if __name__ == "__main__":
    # Veri dosyasının yolunu belirler (bu scriptin bulunduğu klasöre göre ayarlanır)
    file_name = os.path.join(os.path.dirname(__file__), 'datas', 'att48.txt')
    
    # Veriyi yükle ve mesafe matrisini al
    data = load_tsp_data(file_name)
    
    if data is not None:
        print(f"Yüklenen şehir sayısı (N): {len(data)}")
        print("Mesafe matrisi başarıyla oluşturuldu.")
        # İsteğe bağlı: Mesafe matrisinin bir kısmını yazdırma
        # print(dist[:5, :5]) 