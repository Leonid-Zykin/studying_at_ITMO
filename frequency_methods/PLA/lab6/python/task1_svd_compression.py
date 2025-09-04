import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'images', 'task1')
INPUT_IMAGE = os.path.join(IMAGES_DIR, 'original_image.png')


def ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)


def load_grayscale_matrix(path: str) -> np.ndarray:
    img = Image.open(path).convert('L')
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr


def svd_compress(A: np.ndarray, k: int) -> np.ndarray:
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    Uk = U[:, :k]
    Sk = S[:k]
    Vtk = Vt[:k, :]
    Ak = (Uk * Sk) @ Vtk
    Ak = np.clip(Ak, 0.0, 1.0)
    return Ak


def reconstruct_from_svd(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    Uk = U[:, :k]
    Sk = S[:k]
    Vtk = Vt[:k, :]
    Ak = (Uk * Sk) @ Vtk
    return np.clip(Ak, 0.0, 1.0)


def save_image(arr: np.ndarray, path: str) -> None:
    img = Image.fromarray((arr * 255.0).astype(np.uint8))
    img.save(path)


def compute_storage_ratio(m: int, n: int, k: int) -> float:
    # Original: m*n values. Compressed: m*k + k + k*n values
    original = m * n
    compressed = m * k + k + k * n
    return compressed / original


def main():
    ensure_dirs()
    if not os.path.exists(INPUT_IMAGE):
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

    A = load_grayscale_matrix(INPUT_IMAGE)
    m, n = A.shape

    # Choose at least 9 k values spanning the rank range
    max_k = min(m, n)
    k_values = sorted(set([
        max(1, max_k // 100),
        max(2, max_k // 50),
        max(5, max_k // 25),
        max(10, max_k // 20),
        max(15, max_k // 15),
        max(20, max_k // 12),
        max(30, max_k // 10),
        max(40, max_k // 8),
        max(60, max_k // 6),
        max(80, max_k // 5),
        max(100, max_k // 4)
    ]))
    # Ensure presence of very small k
    k_values = sorted(set([k for k in k_values if k <= max_k] + [1, 2, 3]))
    k_values = [k for k in k_values if 1 <= k <= max_k]
    if len(k_values) < 9:
        # fallback to linear spread of 9 values
        k_values = np.linspace(1, max_k, num=9, dtype=int).tolist()
        k_values = sorted(set(k_values + [1, 2, 3]))
    # Make sure we still keep at least 9 after de-dup
    if len(k_values) > 9:
        k_values = k_values[:max(9, len(k_values))]

    # Save original copy for report
    save_image(A, os.path.join(IMAGES_DIR, 'svd_original.png'))

    # Compute SVD once
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    ratios = []
    for k in k_values:
        Ak = reconstruct_from_svd(U, S, Vt, k)
        out_path = os.path.join(IMAGES_DIR, f'svd_k{k}.png')
        save_image(Ak, out_path)
        ratio = compute_storage_ratio(m, n, k)
        ratios.append((k, ratio))
        print(f"k={k}: storage ratio={ratio:.4f}")

    # Plot quality vs k (MSE) and compression ratio
    mses = []
    for k in k_values:
        Ak = reconstruct_from_svd(U, S, Vt, k)
        mse = float(np.mean((A - Ak) ** 2))
        mses.append(mse)

    plt.figure(figsize=(7, 5), dpi=150)
    plt.plot(k_values, mses, marker='o')
    plt.xlabel('k')
    plt.ylabel('MSE')
    plt.title('SVD Compression: Reconstruction Error vs k')
    plt.grid(True, ls=':')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'svd_mse_vs_k.png'))
    plt.close()

    plt.figure(figsize=(7, 5), dpi=150)
    plt.plot([k for k, _ in ratios], [r for _, r in ratios], marker='s')
    plt.xlabel('k')
    plt.ylabel('Compressed/Original storage')
    plt.title('SVD Compression: Storage Ratio vs k')
    plt.grid(True, ls=':')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'svd_storage_ratio_vs_k.png'))
    plt.close()

    # Grid of selected reconstructions (e.g., first 9 ks)
    grid_ks = k_values[:9]
    cols = 3
    rows = int(np.ceil(len(grid_ks) / cols))
    plt.figure(figsize=(12, 4 * rows), dpi=150)
    for idx, k in enumerate(grid_ks, start=1):
        Ak = reconstruct_from_svd(U, S, Vt, k)
        plt.subplot(rows, cols, idx)
        plt.imshow(Ak, cmap='gray', vmin=0, vmax=1)
        plt.title(f'k={k}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'svd_grid.png'))
    plt.close()


if __name__ == '__main__':
    main() 