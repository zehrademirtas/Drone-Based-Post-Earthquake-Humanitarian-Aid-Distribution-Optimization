#Neslihan KARADENİZ Zehra DEMİRTAŞ 21.12.2024
import os
import pandas as pd
import networkx as nx
import heapq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text  # etiket çakışmalarini önler

# OpenMP kullanımı için thread sayısını sınırlama
os.environ["OMP_NUM_THREADS"] = "1"

#excel dosyalarini okuma
def veri_yukleme(mesafeler_path, talepler_path):
    try:
        mesafeler_df = pd.read_excel(mesafeler_path, sheet_name='Mesafeler')
        mesafeler_df.set_index('Kaynak', inplace=True)

        talepler_df = pd.read_excel(talepler_path, sheet_name='Talepler')
        talepler_df.set_index('Nokta', inplace=True)

        print("Veriler başarıyla yüklendi.")
        return mesafeler_df, talepler_df
    except Exception as e:
        print(f"Veri yüklenirken bir hata oluştu: {e}")
        return None, None

#graf olusturma
def graf_olusturma(mesafeler_df):
    G = nx.Graph()
    for kaynak in mesafeler_df.index:
        for hedef, uzaklik in mesafeler_df.loc[kaynak].items():
            if pd.notna(uzaklik):
                G.add_edge(kaynak.strip(), hedef.strip(), weight=uzaklik)
    return G

#k-means algoritmasi
def kmeans_algoritmasi(mesafeler_df, n_clusters):
    # ihtiyac noktalarini alma
    filtered_labels = [label for label in mesafeler_df.index if "Depo" not in label]
    filtered_data = mesafeler_df.loc[filtered_labels, filtered_labels]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_data)
    # k-means kümeleme
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    # PCA  boyut azaltma
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)
    # gorsellestirme
    plt.figure(figsize=(12, 8))
    for cluster_id in range(n_clusters):
        cluster_points = reduced_data[clusters == cluster_id]
        cluster_labels = [label for idx, label in enumerate(filtered_labels) if clusters[idx] == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Küme {cluster_id + 1}")
        for point, label in zip(cluster_points, cluster_labels):
            plt.text(point[0], point[1], label, fontsize=8, ha='right')

    plt.title("K-Means Kümeleme Sonucu")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return clusters, reduced_data, filtered_labels

#dijkstra algoritmasi en kisa mesafe
def dijkstra_algoritmasi(graph, start_node):
    shortest_paths = {node: float('inf') for node in graph.nodes()}
    shortest_paths[start_node] = 0
    priority_queue = [(0, start_node)]  

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_distance > shortest_paths[current_node]:
            continue
        for neighbor, attributes in graph[current_node].items():
            weight = attributes['weight']
            distance = current_distance + weight
            if distance < shortest_paths[neighbor]:
                shortest_paths[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    return shortest_paths

# depolar icin koordinat noktalari olusturma 
def pca_koordinat_olusturma(reduced_data, labels):
    depo_coordinates = {"Depo_Kuzey": [-1, -1], "Depo_Güney": [0, 2]}  #depo koordinatlar noktalari
    pca_df = pd.DataFrame(reduced_data, columns=["PCA1", "PCA2"], index=labels)
    # depo koordinatlarını ekleme
    for depo, coords in depo_coordinates.items():
        if depo not in pca_df.index:
            pca_df.loc[depo] = coords

    return pca_df

#drone rota görsellestirme
def visualize_drone_routes(drones, pca_coordinates_df, cluster_id):
    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    line_style = '--' 
    drone_names = [f"Drone {cluster_id * 3 + i + 1}" for i in range(len(drones))]
    visited_points = {}  #hangi nokta hangi drone

    # drone rotalarini cizme
    for i, drone in enumerate(drones):
        route = drone["route"]
        color = colors[i % len(colors)]
        for j in range(len(route) - 1):
            start, end = route[j], route[j + 1]
            if start in pca_coordinates_df.index and end in pca_coordinates_df.index:
                start_x, start_y = pca_coordinates_df.loc[start, "PCA1"], pca_coordinates_df.loc[start, "PCA2"]
                end_x, end_y = pca_coordinates_df.loc[end, "PCA1"], pca_coordinates_df.loc[end, "PCA2"]
                plt.plot(
                    [start_x, end_x], [start_y, end_y],
                    color=color, linestyle=line_style, linewidth=2, alpha=0.8,
                    label=drone_names[i] if j == 0 else ""  # Legend için sadece bir kez etiket eklenir
                )
                # gidilen noktayi kaydet 
                if "Depo" not in end:
                    visited_points[end] = color
    # depo görsel
    for depot in ["Depo_Kuzey", "Depo_Güney"]:
        if depot in pca_coordinates_df.index:
            x, y = pca_coordinates_df.loc[depot, "PCA1"], pca_coordinates_df.loc[depot, "PCA2"]
            plt.scatter(x, y, color='red', marker='s', s=200, edgecolors='black')
            plt.text(x + 0.1, y, depot, fontsize=12, ha='left', va='center', color='black', fontweight='bold')

    # ihtiyac noktalarini goster
    for point, (x, y) in pca_coordinates_df.iterrows():
        if "Depo" not in point:  
            point_color = visited_points.get(point, 'black')  #gidilmediyse siyah
            plt.scatter(x, y, color=point_color, s=50)
            plt.text(x, y, point, fontsize=9, ha='center', va='bottom', color=point_color)

    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1), title="Drone Renkleri")

    # görsellestirme
    plt.title(f"Küme {cluster_id + 1} için Drone Rotaları", fontsize=14, fontweight='bold')
    plt.xlabel("PCA 1", fontsize=12)
    plt.ylabel("PCA 2", fontsize=12)
    plt.tight_layout()
    plt.grid(True)
    plt.show()

# her drone nun rota görseli
def visualize_multiple_drone_routes(drones, pca_coordinates_df, cluster_id):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    sns.set_style("whitegrid")
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    line_style = '--'  

    for i, drone in enumerate(drones):
        ax = axes[i]
        route = drone["route"]
        color = colors[i % len(colors)]
        visited_points = set()  #gidilen ihitac noktalari

        # rota cizimi
        for j in range(len(route) - 1):
            start, end = route[j], route[j + 1]
            if start in pca_coordinates_df.index and end in pca_coordinates_df.index:
                start_x, start_y = pca_coordinates_df.loc[start, "PCA1"], pca_coordinates_df.loc[start, "PCA2"]
                end_x, end_y = pca_coordinates_df.loc[end, "PCA1"], pca_coordinates_df.loc[end, "PCA2"]
                ax.plot(
                    [start_x, end_x], [start_y, end_y],
                    color=color, linestyle=line_style, linewidth=2, alpha=0.8
                )
                if "Depo" not in end:  # ihtiyac noktasi ise ekle
                    visited_points.add(end)

        # Depo noktalarini vurgula
        for depot in ["Depo_Kuzey", "Depo_Güney"]:
            if depot in pca_coordinates_df.index:
                x, y = pca_coordinates_df.loc[depot, "PCA1"], pca_coordinates_df.loc[depot, "PCA2"]
                ax.scatter(x, y, color='red', marker='s', s=200, edgecolors='black')
                ax.text(x + 0.1, y, depot, fontsize=10, ha='left', va='center', color='black', fontweight='bold')

        # gidilen noktalari gösterme
        for point in visited_points:
            x, y = pca_coordinates_df.loc[point, "PCA1"], pca_coordinates_df.loc[point, "PCA2"]
            ax.scatter(x, y, color=color, s=50)
            ax.text(x, y, point, fontsize=6, ha='center', va='bottom', color='black') 

        # görsellestirme
        ax.set_title(f"{drone['name']} Rotaları", fontsize=12, fontweight='bold')
        ax.set_xlabel("PCA 1", fontsize=10)
        ax.grid(True)

    axes[0].set_ylabel("PCA 2", fontsize=10)
    fig.suptitle(f"Küme {cluster_id + 1} için Drone Rotaları", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()

# sefer ve mesafe grafikleri
def visualize_overall_drone_performance(all_drones):
    drone_names = [drone["name"] for drone in all_drones]
    total_distances = [drone["distance"] for drone in all_drones]
    total_trips = [drone["trips"] for drone in all_drones]

    # yan yana toplam mesafe ve sefer grafikleri
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # mesafe grafigi
    bars = axes[0].bar(drone_names, total_distances, color='skyblue', label='Toplam Mesafe (km)')
    axes[0].set_title("Tüm Dronlar İçin Toplam Mesafeler")
    axes[0].set_ylabel("Mesafe (km)")
    axes[0].set_xlabel("Drone Adı")
    axes[0].tick_params(axis='x', rotation=45)

    # degerleri cubuk uzerine yazma
    for bar, value in zip(bars, total_distances):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.1f}', 
                     ha='center', va='bottom', fontsize=8)

    # sefer sayisi grafigi
    bars = axes[1].bar(drone_names, total_trips, color='green', label='Toplam Sefer Sayısı')
    axes[1].set_title("Tüm Dronlar İçin Toplam Sefer Sayıları")
    axes[1].set_ylabel("Sefer Sayısı")
    axes[1].set_xlabel("Drone Adı")
    axes[1].tick_params(axis='x', rotation=45)

    # degerleri cubuk uzerine yazma
    for bar, value in zip(bars, total_trips):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value}', 
                     ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

# sefer tablosu
def visualize_cluster_delivery_tables(cluster_id, drone_logs):
    if not drone_logs:
        print(f"Küme {cluster_id} için veri bulunamadı.")
        return

    try:
        df = pd.DataFrame(drone_logs)
        #sutun isimleri
        expected_columns = ["Drone", "Depo", "Hedef Nokta", "Tibbi Malzeme", "Yiyecek", "Doluluk Oranı (%)", "Mesafe (km)"]
        df = df[expected_columns]

        # tabloyu görsellestirme
        plt.figure(figsize=(10, len(df) // 2 + 1))
        plt.axis("off")
        table = plt.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
        table.scale(1, 2) 
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns))))
        plt.title(f"Küme {cluster_id} için Drone Yükleme Planı", fontsize=14, weight='bold', loc='center')
        plt.show()
    except Exception as e:
        print(f"Hata oluştu: {e}")

# kume noktalari arasi mesafeler isi haritası
def isi_haritasi(mesafeler_df, clusters, labels, n_clusters):
    for cluster_id in range(n_clusters):
        # kume noktalari secme
        cluster_indices = [i for i, label in enumerate(labels) if clusters[i] == cluster_id]
        cluster_labels = [labels[i] for i in cluster_indices]  
        
        cluster_distances = mesafeler_df.loc[cluster_labels, cluster_labels]
        
        # isi haritasini cizme
        plt.figure(figsize=(10, 8))
        sns.heatmap(cluster_distances, annot=True, fmt=".1f", cmap="viridis", cbar_kws={'label': 'Mesafe (km)'})
        plt.title(f"Küme {cluster_id + 1} - İhtiyaç Noktaları Arasındaki Mesafeler Isı Haritası")
        plt.xlabel("Noktalar")
        plt.ylabel("Noktalar")
        plt.tight_layout()
        plt.show()

# her kume icin 3 drone yukleme plani
def ddrone_yukleme_plani(talepler_df, mesafeler_df, clusters, filtered_labels, drone_capacity):
    all_drones = []  # dron listesi
    overall_total_distance = 0  # toplam mesafe
    overall_total_trips = 0  #  toplam sefer sayisi
    drone_logs = [] 

    for cluster_id in set(clusters):
        print(f"\n--- Küme {cluster_id + 1} için Drone Planı ---")

        # kume noktalarini al
        cluster_points = [label for idx, label in enumerate(filtered_labels) if clusters[idx] == cluster_id]

        if len(cluster_points) == 0:
            print(f"Küme {cluster_id + 1} boş veya taleplerde yer almıyor, drone planı oluşturulmadı.")
            continue

        cluster_total_distance = 0  # kumenin toplam mesafesi
        remaining_demands = talepler_df.loc[cluster_points].copy()
        drones = [{"name": f"Drone {cluster_id * 3 + i + 1}", "distance": 0, "trips": 0, "route": []} for i in range(3)]
        current_drone_idx = 0  

        cluster_data = []  #  drone ve rota verileri

        for nokta in remaining_demands.index:
            # en yakin depoyu bul 
            depo_mesafeleri = {
                "Depo_Kuzey": mesafeler_df.at["Depo_Kuzey", nokta],
                "Depo_Güney": mesafeler_df.at["Depo_Güney", nokta],
            }
            en_yakin_depo = min(depo_mesafeleri, key=depo_mesafeleri.get)

            while remaining_demands.at[nokta, "Tibbi_Malzeme"] > 0 or remaining_demands.at[nokta, "Yiyecek"] > 0:
                current_drone = drones[current_drone_idx]
                current_load = 0

                # yukleme yap
                tibbi = min(drone_capacity - current_load, remaining_demands.at[nokta, "Tibbi_Malzeme"])
                yiyecek = min(drone_capacity - current_load - tibbi, remaining_demands.at[nokta, "Yiyecek"])
                total_load = tibbi + yiyecek

                if total_load == 0:
                    break

                # en yakin depodan mesafeyi hesapla (gidis-donus)
                distance_to_nokta = depo_mesafeleri[en_yakin_depo] * 2
                current_drone["distance"] += distance_to_nokta
                cluster_total_distance += distance_to_nokta
                current_drone["trips"] += 1

                # Rotaya depo -> nokta -> depo ekle
                current_drone["route"].extend([en_yakin_depo, nokta, en_yakin_depo])

                # talepleri guncelle
                remaining_demands.at[nokta, "Tibbi_Malzeme"] -= tibbi
                remaining_demands.at[nokta, "Yiyecek"] -= yiyecek

                drone_logs.append({
                    "Drone": current_drone["name"],
                    "Depo": en_yakin_depo,
                    "Hedef Nokta": nokta,
                    "Tibbi Malzeme": tibbi,
                    "Yiyecek": yiyecek,
                    "Doluluk Oranı (%)": (total_load / drone_capacity) * 100,
                    "Mesafe (km)": distance_to_nokta,
                    "Cluster": cluster_id,  
                })

                #gorsellestirme veri
                cluster_data.append({
                    "drone": current_drone["name"],
                    "depot": en_yakin_depo,
                    "target": nokta,
                    "tibbi": tibbi,
                    "yiyecek": yiyecek
                })

                current_drone_idx = (current_drone_idx + 1) % 3

        # drone bilgilerini yazdirma
        for drone in drones:
            print(f"{drone['name']}: Toplam {drone['trips']} sefer, "
                  f"Gidilen Toplam Mesafe: {drone['distance']:.2f} km, "
                  f"\nRota: {' -> '.join(drone['route'])}")

        all_drones.extend(drones)

        print(f"Küme {cluster_id + 1} için Toplam Mesafe: {cluster_total_distance:.2f} km")
        overall_total_distance += cluster_total_distance
        overall_total_trips += sum(drone["trips"] for drone in drones)

        #gorsellestirme fonk
        drone_plan_gorsel(cluster_data, pca_koordinat_olusturma(reduced_data, filtered_labels))
        visualize_multiple_drone_routes(drones, pca_koordinat_olusturma(reduced_data, filtered_labels), cluster_id)

    # her kume icin tablo
    for cluster_id in set(clusters):
        visualize_cluster_delivery_tables(cluster_id + 1, [log for log in drone_logs if log["Cluster"] == cluster_id])

    print("\n--- Tüm Drone Performansı ---")
    visualize_overall_drone_performance(all_drones)
    print(f"Tüm Dronlar İçin Genel Toplam Mesafe: {overall_total_distance:.2f} km")
    print(f"Tüm Dronlar İçin Genel Toplam Sefer Sayısı: {overall_total_trips}")

#drone plan gorsellestirme
def drone_plan_gorsel(cluster_data, pca_coordinates_df):
    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    line_style = '--'

    for i, entry in enumerate(cluster_data):
        drone_name = entry["drone"]
        start = entry["depot"]
        end = entry["target"]
        color = colors[i % len(colors)]

        # cizgiyle rota gosterimi
        if start in pca_coordinates_df.index and end in pca_coordinates_df.index:
            start_x, start_y = pca_coordinates_df.loc[start, "PCA1"], pca_coordinates_df.loc[start, "PCA2"]
            end_x, end_y = pca_coordinates_df.loc[end, "PCA1"], pca_coordinates_df.loc[end, "PCA2"]
            plt.plot(
                [start_x, end_x], [start_y, end_y],
                color=color, linestyle=line_style, linewidth=2, alpha=0.8
            )

        # depolari vurgulama
        plt.scatter(*pca_coordinates_df.loc[start], color='red', marker='s', s=200, edgecolors='black')
        plt.text(*pca_coordinates_df.loc[start], start, fontsize=12, ha='left', va='center', fontweight='bold', color='black')

        # bitis noktalari
        plt.scatter(*pca_coordinates_df.loc[end], color=color, s=100, edgecolors='black')
        plt.text(*pca_coordinates_df.loc[end], end, fontsize=10, ha='center', va='bottom', color='black')

    #grafik
    plt.title("Küme Planı - Drone Rotası", fontsize=14, fontweight='bold')
    plt.xlabel("PCA 1", fontsize=12)
    plt.ylabel("PCA 2", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    mesafeler_path = "Ihtiyac_Noktalari.xlsx"
    talepler_path = "Yardim_Talepleri.xlsx"
    drone_capacity = 30
    n_clusters = 3

    mesafeler_df, talepler_df = veri_yukleme(mesafeler_path, talepler_path)

    if mesafeler_df is not None and talepler_df is not None:
        # KMeans Kumeleme
        clusters, reduced_data, filtered_labels = kmeans_algoritmasi(mesafeler_df, n_clusters)
        #PCA koordinat
        pca_coordinates_df = pca_koordinat_olusturma(reduced_data, filtered_labels)
        # graf olusturma
        G = graf_olusturma(mesafeler_df)
        #kume isi haritasi
        isi_haritasi(mesafeler_df, clusters, filtered_labels, n_clusters)
        # drone dagitim plani
        ddrone_yukleme_plani(talepler_df, mesafeler_df, clusters, filtered_labels, drone_capacity)
    else:
        print("Hata: Veriler yüklenemedi.")