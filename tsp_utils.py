import pandas as pd
import numpy as np
import os
# import matplotlib.pyplot as plt

# Sonuçları (parametreler ve fitness değeri) bir Excel dosyasına kaydeder
def save_results_to_excel(results_list, output_filename='results.xlsx'):
    try:
        # Create DataFrame from new results
        new_df = pd.DataFrame(results_list, columns=['ER', 'CR', 'MR', 'Final Distance'])
        
        # Add timestamp column
        from datetime import datetime
        new_df['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if file exists
        if os.path.exists(output_filename):
            # Read existing file
            existing_df = pd.read_excel(output_filename)
            # Concatenate with new results
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        # Save combined results
        combined_df.to_excel(output_filename, index=False)
        print(f"Results appended to {output_filename}")
    except Exception as e:
        print(f"Error saving results to {output_filename}: {e}")

# Şehirleri ve rotayı çizen fonksiyon (şu anda kullanılmıyor, gelecekte etkinleştirilebilir)
# def plot_tsp_solution(cities, path, distance, it, filename=None):
#     # Plotting code here
#     pass

# Orijinal plot_cities fonksiyonunun taşınmış hali (yorum satırı)
# def plot_cities(cities, it, path, distance, fig, ax):
#     ax.clear()
#     ax.scatter(cities[:, 1], cities[:, 2], c='red', label='Cities', zorder=5)
#     for i, (x, y) in enumerate(cities[:, 1:3]):
#         ax.text(x, y, str(i), color="black", fontsize=9, zorder=5)
#     if path is not None:
#         path = np.array(path, dtype=int)
#         path_coords = cities[path, 1:3]
#         ax.plot(path_coords[:, 0], path_coords[:, 1], 'k-', lw=1, zorder=1)
#         ax.plot([path_coords[-1, 0], path_coords[0, 0]], [path_coords[-1, 1], path_coords[0, 1]], 'k-', lw=1, label='Path', zorder=1)
#     ax.set_xlabel('X Coordinate')
#     ax.set_ylabel('Y Coordinate')
#     ax.set_title(f'Iteration: {it} Total Distance: {distance:.2f}')
#     ax.legend() 