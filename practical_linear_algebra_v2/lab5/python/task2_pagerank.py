import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

# Создаем папку для сохранения изображений
os.makedirs('../images/task2', exist_ok=True)

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Создаем ориентированный граф для PageRank
def create_web_graph():
    """Создание ориентированного графа веб-страниц"""
    
    # Создаем ориентированный граф
    G = nx.DiGraph()
    
    # Добавляем вершины (10-15 веб-страниц)
    n_vertices = 12
    G.add_nodes_from(range(1, n_vertices + 1))
    
    # Определяем связи между страницами (25-50 стрелок)
    edges = [
        # Связи от страницы 1
        (1, 2), (1, 3), (1, 4),
        
        # Связи от страницы 2
        (2, 1), (2, 3), (2, 5), (2, 6),
        
        # Связи от страницы 3
        (3, 1), (3, 2), (3, 4), (3, 7),
        
        # Связи от страницы 4
        (4, 1), (4, 3), (4, 8), (4, 9),
        
        # Связи от страницы 5
        (5, 2), (5, 6), (5, 10),
        
        # Связи от страницы 6
        (6, 2), (6, 5), (6, 7), (6, 11),
        
        # Связи от страницы 7
        (7, 3), (7, 6), (7, 8), (7, 12),
        
        # Связи от страницы 8
        (8, 4), (8, 7), (8, 9),
        
        # Связи от страницы 9
        (9, 4), (9, 8), (9, 10), (9, 12),
        
        # Связи от страницы 10
        (10, 5), (10, 9), (10, 11),
        
        # Связи от страницы 11
        (11, 6), (11, 10), (11, 12),
        
        # Связи от страницы 12
        (12, 7), (12, 9), (12, 11)
    ]
    
    G.add_edges_from(edges)
    
    return G

# Функция для построения матрицы переходов
def build_transition_matrix(G):
    """Построение матрицы переходов M для PageRank"""
    
    n = len(G.nodes())
    M = np.zeros((n, n))
    
    for j in range(1, n + 1):
        # Находим все исходящие связи из вершины j
        out_edges = list(G.out_edges(j))
        out_degree = len(out_edges)
        
        if out_degree > 0:
            for edge in out_edges:
                i = edge[1]  # Конечная вершина
                M[i-1, j-1] = 1.0 / out_degree  # mij = 1 / степень исхода j
    
    return M

# Функция для вычисления PageRank
def compute_pagerank(M, d=1.0, max_iter=1000, tol=1e-6):
    """Вычисление PageRank с использованием степенного метода"""
    
    n = M.shape[0]
    
    # Начальный вектор PageRank (равномерное распределение)
    p = np.ones(n) / n
    
    # Итерационный процесс
    for iteration in range(max_iter):
        p_new = d * M @ p + (1 - d) / n
        
        # Проверяем сходимость
        if np.linalg.norm(p_new - p) < tol:
            print(f"PageRank сошелся за {iteration + 1} итераций")
            break
        
        p = p_new
    
    return p

# Функция для визуализации графа с PageRank
def plot_graph_with_pagerank(G, pagerank, title, filename):
    """Визуализация графа с размерами узлов пропорциональными PageRank"""
    
    plt.figure(figsize=(12, 8))
    
    # Позиции вершин
    pos = nx.spring_layout(G, seed=42)
    
    # Нормализуем PageRank для отображения размеров узлов
    node_sizes = 1000 + 2000 * pagerank / np.max(pagerank)
    
    # Рисуем граф
    nx.draw(G, pos,
            node_size=node_sizes,
            node_color='lightblue',
            with_labels=True,
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            width=1.5,
            arrows=True,
            arrowsize=20,
            arrowstyle='->')
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'../images/task2/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Функция для анализа собственных чисел матрицы M
def analyze_eigenvalues(M, title, filename):
    """Анализ собственных чисел матрицы переходов"""
    
    eigenvalues, eigenvectors = np.linalg.eig(M)
    
    # Сортируем по убыванию модуля
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    
    plt.figure(figsize=(12, 5))
    
    # График собственных чисел
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(sorted_eigenvalues) + 1), np.real(sorted_eigenvalues), 'bo-', markersize=8)
    plt.xlabel('Индекс')
    plt.ylabel('Re(λ)')
    plt.title('Вещественные части собственных чисел')
    plt.grid(True, alpha=0.3)
    
    # График модулей собственных чисел
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(sorted_eigenvalues) + 1), np.abs(sorted_eigenvalues), 'ro-', markersize=8)
    plt.xlabel('Индекс')
    plt.ylabel('|λ|')
    plt.title('Модули собственных чисел')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'../images/task2/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return eigenvalues, eigenvectors

# Функция для анализа сходимости PageRank
def analyze_convergence(M, d_values=[0.85, 0.9, 0.95, 1.0]):
    """Анализ сходимости PageRank для разных значений d"""
    
    plt.figure(figsize=(15, 10))
    
    for i, d in enumerate(d_values):
        print(f"\n--- Анализ для d = {d} ---")
        
        n = M.shape[0]
        p = np.ones(n) / n  # Начальный вектор
        
        # Отслеживаем изменения
        changes = []
        iterations = []
        
        for iteration in range(100):
            p_new = d * M @ p + (1 - d) / n
            change = np.linalg.norm(p_new - p)
            changes.append(change)
            iterations.append(iteration)
            
            if change < 1e-6:
                print(f"Сходимость достигнута за {iteration + 1} итераций")
                break
            
            p = p_new
        
        # График сходимости
        plt.subplot(2, 2, i+1)
        plt.semilogy(iterations, changes, 'o-', markersize=4, label=f'd = {d}')
        plt.xlabel('Итерация')
        plt.ylabel('Изменение (log scale)')
        plt.title(f'Сходимость PageRank, d = {d}')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('../images/task2/pagerank_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()

# Основная функция
def main():
    print("Создание веб-графа...")
    
    # Создаем граф
    G = create_web_graph()
    
    print(f"Создан ориентированный граф с {len(G.nodes())} вершинами и {len(G.edges())} рёбрами")
    print(f"Связи между страницами:")
    for edge in G.edges():
        print(f"  Страница {edge[0]} -> Страница {edge[1]}")
    
    # Визуализируем исходный граф
    plot_graph_with_pagerank(G, np.ones(len(G.nodes())), 
                           "Исходный веб-граф", "original_web_graph")
    
    # Строим матрицу переходов
    print(f"\nПостроение матрицы переходов...")
    M = build_transition_matrix(G)
    
    print(f"Матрица переходов M:")
    print(M)
    
    # Проверяем свойства матрицы M
    print(f"\nСвойства матрицы M:")
    print(f"Размер: {M.shape}")
    print(f"Сумма по столбцам (должна быть 1 или 0):")
    for j in range(M.shape[1]):
        col_sum = np.sum(M[:, j])
        print(f"  Столбец {j+1}: {col_sum:.4f}")
    
    # Анализируем собственные числа
    print(f"\nАнализ собственных чисел матрицы M...")
    eigenvalues, eigenvectors = analyze_eigenvalues(M, "Собственные числа матрицы переходов", "eigenvalues")
    
    print(f"Собственные числа:")
    for i, val in enumerate(eigenvalues[:5]):
        print(f"  λ_{i+1} = {val:.4f}")
    
    # Находим собственный вектор, соответствующий наибольшему собственному числу
    max_eigenvalue_idx = np.argmax(np.abs(eigenvalues))
    max_eigenvalue = eigenvalues[max_eigenvalue_idx]
    max_eigenvector = eigenvectors[:, max_eigenvalue_idx]
    
    print(f"\nНаибольшее собственное число: λ = {max_eigenvalue:.4f}")
    print(f"Соответствующий собственный вектор:")
    for i, val in enumerate(max_eigenvector):
        print(f"  v_{i+1} = {val:.4f}")
    
    # Вычисляем PageRank
    print(f"\nВычисление PageRank...")
    pagerank = compute_pagerank(M, d=1.0)
    
    print(f"PageRank значения:")
    for i, pr in enumerate(pagerank):
        print(f"  Страница {i+1}: {pr:.4f}")
    
    # Ранжируем страницы по PageRank
    ranking = np.argsort(pagerank)[::-1]
    print(f"\nРанжирование страниц по PageRank:")
    for i, page_idx in enumerate(ranking):
        print(f"  {i+1}. Страница {page_idx+1}: {pagerank[page_idx]:.4f}")
    
    # Визуализируем результат
    plot_graph_with_pagerank(G, pagerank, 
                           "Веб-граф с PageRank (размер узла ~ PageRank)", 
                           "pagerank_result")
    
    # Анализируем сходимость для разных значений d
    print(f"\nАнализ сходимости для разных значений d...")
    analyze_convergence(M)
    
    # Сравниваем PageRank для разных значений d
    d_values = [0.85, 0.9, 0.95, 1.0]
    pagerank_results = {}
    
    plt.figure(figsize=(12, 8))
    
    for i, d in enumerate(d_values):
        pagerank_d = compute_pagerank(M, d=d)
        pagerank_results[d] = pagerank_d
        
        plt.subplot(2, 2, i+1)
        pages = range(1, len(pagerank_d) + 1)
        plt.bar(pages, pagerank_d, alpha=0.7)
        plt.xlabel('Страница')
        plt.ylabel('PageRank')
        plt.title(f'PageRank для d = {d}')
        plt.xticks(pages)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/task2/pagerank_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Анализ интерпретации
    print(f"\nИнтерпретация результатов:")
    print(f"1. Матрица M представляет вероятности перехода между страницами")
    print(f"2. Собственный вектор с наибольшим собственным числом = 1 представляет стационарное распределение")
    print(f"3. PageRank показывает 'важность' каждой страницы в сети")
    print(f"4. Параметр d (damping factor) контролирует вероятность случайного перехода")
    print(f"5. При d = 1 алгоритм соответствует марковскому процессу без затухания")
    
    print("\nPageRank алгоритм завершен!")
    print("Все графики сохранены в папке images/task2/")

if __name__ == "__main__":
    main() 