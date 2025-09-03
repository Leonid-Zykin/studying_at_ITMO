import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'images', 'task1')
INPUT_IMAGE = os.path.join(IMAGES_DIR, 'original_image.png')


def load_grayscale_matrix(path: str) -> np.ndarray:
    img = Image.open(path).convert('L')
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr


def svd_reconstructors(A: np.ndarray):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U, S, Vt


def svd_compress_from(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    Uk = U[:, :k]
    Sk = S[:k]
    Vtk = Vt[:k, :]
    Ak = (Uk * Sk) @ Vtk
    Ak = np.clip(Ak, 0.0, 1.0)
    return Ak


def compute_storage_ratio(m: int, n: int, k: int) -> float:
    original = m * n
    compressed = m * k + k + k * n
    return compressed / original


def main():
    if not os.path.exists(INPUT_IMAGE):
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

    A = load_grayscale_matrix(INPUT_IMAGE)
    m, n = A.shape

    # Detect available k-files if present, otherwise pick a default spread
    k_files = []
    pattern = re.compile(r"svd_k(\d+)\.png$")
    for name in os.listdir(IMAGES_DIR):
        mobj = pattern.match(name)
        if mobj:
            k_files.append(int(mobj.group(1)))
    if k_files:
        k_values = sorted(set(k_files))
    else:
        max_k = min(m, n)
        k_values = np.linspace(1, max_k, num=9, dtype=int).tolist()

    U, S, Vt = svd_reconstructors(A)

    # MSE and storage ratio
    mses = []
    ratios = []
    for k in k_values:
        Ak = svd_compress_from(U, S, Vt, k)
        mses.append(float(np.mean((A - Ak) ** 2)))
        ratios.append(compute_storage_ratio(m, n, k))

    # Plots
    os.makedirs(IMAGES_DIR, exist_ok=True)

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
    plt.plot(k_values, ratios, marker='s')
    plt.xlabel('k')
    plt.ylabel('Compressed/Original storage')
    plt.title('SVD Compression: Storage Ratio vs k')
    plt.grid(True, ls=':')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'svd_storage_ratio_vs_k.png'))
    plt.close()

    # Grid for first 9 k's
    grid_ks = k_values[:9]
    cols = 3
    rows = int(np.ceil(len(grid_ks) / cols))
    plt.figure(figsize=(12, 4 * rows), dpi=150)
    for idx, k in enumerate(grid_ks, start=1):
        Ak = svd_compress_from(U, S, Vt, k)
        plt.subplot(rows, cols, idx)
        plt.imshow(Ak, cmap='gray', vmin=0, vmax=1)
        plt.title(f'k={k}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'svd_grid.png'))
    plt.close()


if __name__ == '__main__':
    main() 