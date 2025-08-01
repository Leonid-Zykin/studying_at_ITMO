import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# Создаем папку для сохранения изображений
os.makedirs('../images/task1', exist_ok=True)

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Создаем социальную сеть с явными сообществами
def create_social_network():
    """Создание социальной сети с тремя сообществами"""
    
    # Создаем граф
    G = nx.Graph()
    
    # Добавляем вершины (15-30 человек)
    n_vertices = 20
    G.add_nodes_from(range(1, n_vertices + 1))
    
    # Определяем сообщества
    community1 = [1, 2, 3, 4, 5, 6, 7]      # 7 человек
    community2 = [8, 9, 10, 11, 12, 13]      # 6 человек  
    community3 = [14, 15, 16, 17, 18, 19, 20] # 7 человек
    
    # Добавляем связи внутри сообществ (плотные связи)
    for i in community1:
        for j in community1:
            if i < j and np.random.random() < 0.7:  # 70% вероятность связи
                G.add_edge(i, j)
    
    for i in community2:
        for j in community2:
            if i < j and np.random.random() < 0.7:
                G.add_edge(i, j)
    
    for i in community3:
        for j in community3:
            if i < j and np.random.random() < 0.7:
                G.add_edge(i, j)
    
    # Добавляем несколько связей между сообществами (слабые связи)
    cross_connections = [
        (3, 8), (6, 9), (7, 10),  # между сообществами 1 и 2
        (8, 14), (11, 15), (13, 16),  # между сообществами 2 и 3
        (5, 17), (2, 19)  # между сообществами 1 и 3
    ]
    
    for edge in cross_connections:
        G.add_edge(edge[0], edge[1])
    
    return G, [community1, community2, community3]

# Функция для построения матрицы Лапласа
def laplacian_matrix(G):
    """Построение матрицы Лапласа графа"""
    n = len(G.nodes())
    L = np.zeros((n, n))
    
    # Матрица смежности
    A = nx.adjacency_matrix(G).toarray()
    
    # Диагональная матрица степеней
    D = np.diag([G.degree(i) for i in range(1, n + 1)])
    
    # Матрица Лапласа: L = D - A
    L = D - A
    
    return L

# Функция для спектральной кластеризации
def spectral_clustering(G, k):
    """Спектральная кластеризация графа"""
    
    # Строим матрицу Лапласа
    L = laplacian_matrix(G)
    
    # Находим собственные числа и векторы
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Берем k собственных векторов, соответствующих наименьшим собственным числам
    # (исключаем первое собственное число, которое всегда равно 0)
    indices = np.argsort(eigenvalues)[1:k+1]
    V = eigenvectors[:, indices]
    
    # Применяем k-means к строкам матрицы V
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(V)
    
    return clusters, eigenvalues, eigenvectors, V

# Функция для визуализации графа с кластеризацией
def plot_graph_with_clusters(G, clusters, title, filename):
    """Визуализация графа с раскрашенными кластерами"""
    
    plt.figure(figsize=(12, 8))
    
    # Позиции вершин
    pos = nx.spring_layout(G, seed=42)
    
    # Цвета для кластеров
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Раскрашиваем вершины по кластерам
    node_colors = [colors[clusters[i-1]] for i in G.nodes()]
    
    # Рисуем граф
    nx.draw(G, pos, 
            node_color=node_colors,
            node_size=500,
            with_labels=True,
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            width=1.5)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'../images/task1/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Функция для анализа собственных чисел
def plot_eigenvalues(eigenvalues, title, filename):
    """График собственных чисел"""
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', markersize=8)
    plt.xlabel('Индекс')
    plt.ylabel('Собственное число')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'../images/task1/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Функция для анализа качества кластеризации
def analyze_clustering_quality(G, clusters, k):
    """Анализ качества кластеризации"""
    
    # Вычисляем матрицу Лапласа
    L = laplacian_matrix(G)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Берем k собственных векторов
    indices = np.argsort(eigenvalues)[1:k+1]
    V = eigenvectors[:, indices]
    
    # Вычисляем silhouette score
    silhouette_avg = silhouette_score(V, clusters)
    
    # Вычисляем количество связей внутри кластеров и между кластерами
    internal_edges = 0
    external_edges = 0
    
    for edge in G.edges():
        i, j = edge[0] - 1, edge[1] - 1  # Индексы начинаются с 0
        if clusters[i] == clusters[j]:
            internal_edges += 1
        else:
            external_edges += 1
    
    total_edges = internal_edges + external_edges
    internal_ratio = internal_edges / total_edges if total_edges > 0 else 0
    
    return silhouette_avg, internal_ratio, internal_edges, external_edges

# Основная функция
def main():
    print("Создание социальной сети...")
    
    # Создаем социальную сеть
    G, true_communities = create_social_network()
    
    print(f"Создан граф с {len(G.nodes())} вершинами и {len(G.edges())} рёбрами")
    print(f"Истинные сообщества: {true_communities}")
    
    # Визуализируем исходный граф
    plot_graph_with_clusters(G, [0] * len(G.nodes()), 
                           "Исходная социальная сеть", "original_network")
    
    # Анализируем собственные числа матрицы Лапласа
    L = laplacian_matrix(G)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    print(f"\nСобственные числа матрицы Лапласа:")
    for i, val in enumerate(eigenvalues[:10]):
        print(f"λ_{i+1} = {val:.4f}")
    
    plot_eigenvalues(eigenvalues, "Собственные числа матрицы Лапласа", "eigenvalues")
    
    # Тестируем кластеризацию для разных значений k
    k_values = [2, 3, 4, 5, 6]
    results = []
    
    print(f"\nАнализ кластеризации для разных значений k:")
    
    for k in k_values:
        print(f"\n--- k = {k} ---")
        
        # Выполняем спектральную кластеризацию
        clusters, eigenvalues, eigenvectors, V = spectral_clustering(G, k)
        
        # Анализируем качество
        silhouette_avg, internal_ratio, internal_edges, external_edges = analyze_clustering_quality(G, clusters, k)
        
        results.append({
            'k': k,
            'clusters': clusters,
            'silhouette': silhouette_avg,
            'internal_ratio': internal_ratio,
            'internal_edges': internal_edges,
            'external_edges': external_edges
        })
        
        print(f"Silhouette score: {silhouette_avg:.4f}")
        print(f"Доля внутренних связей: {internal_ratio:.4f}")
        print(f"Внутренние связи: {internal_edges}, внешние связи: {external_edges}")
        
        # Визуализируем результат
        plot_graph_with_clusters(G, clusters, 
                               f"Спектральная кластеризация, k = {k}", 
                               f"clustering_k{k}")
    
    # Сравнительный анализ
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    k_vals = [r['k'] for r in results]
    silhouette_scores = [r['silhouette'] for r in results]
    plt.plot(k_vals, silhouette_scores, 'bo-', markersize=8)
    plt.xlabel('Количество кластеров (k)')
    plt.ylabel('Silhouette score')
    plt.title('Качество кластеризации')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    internal_ratios = [r['internal_ratio'] for r in results]
    plt.plot(k_vals, internal_ratios, 'ro-', markersize=8)
    plt.xlabel('Количество кластеров (k)')
    plt.ylabel('Доля внутренних связей')
    plt.title('Качество разделения сообществ')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/task1/clustering_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Находим оптимальное k
    best_k = k_vals[np.argmax(silhouette_scores)]
    print(f"\nОптимальное количество кластеров: k = {best_k}")
    
    # Детальный анализ для оптимального k
    best_result = results[np.argmax(silhouette_scores)]
    print(f"Лучший silhouette score: {best_result['silhouette']:.4f}")
    print(f"Лучшая доля внутренних связей: {best_result['internal_ratio']:.4f}")
    
    # Анализ собственных векторов
    print(f"\nАнализ собственных векторов для k = {best_k}:")
    L = laplacian_matrix(G)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    indices = np.argsort(eigenvalues)[1:best_k+1]
    
    for i, idx in enumerate(indices):
        print(f"Собственный вектор {i+1} (λ = {eigenvalues[idx]:.4f}):")
        print(f"  Компоненты: {eigenvectors[:, idx][:10]}...")  # Показываем первые 10 компонент
    
    print("\nСпектральная кластеризация завершена!")
    print("Все графики сохранены в папке images/task1/")

if __name__ == "__main__":
    main() 