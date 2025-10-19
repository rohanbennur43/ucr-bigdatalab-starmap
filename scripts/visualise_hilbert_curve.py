import numpy as np
import matplotlib.pyplot as plt
import argparse

def hilbert_xy_to_index(x, y, bits):
    x = x.astype(np.uint64).copy()
    y = y.astype(np.uint64).copy()
    d = np.zeros_like(x, dtype=np.uint64)

    for i in range(bits - 1, -1, -1):
        rx = (x >> i) & 1
        ry = (y >> i) & 1

        d |= (((3 * rx) ^ ry) << (2 * i))

        lowmask = (np.uint64(1) << i) - 1
        xl = x & lowmask
        yl = y & lowmask

        swap = (ry == 0)
        reflect = swap & (rx == 1)

        xl = np.where(reflect, lowmask - xl, xl)
        yl = np.where(reflect, lowmask - yl, yl)

        xt = np.where(swap, yl, xl)
        yt = np.where(swap, xl, yl)

        x = (x & ~lowmask) | xt
        y = (y & ~lowmask) | yt

    return d


def plot_hilbert(bits=4):
    """Visualize Hilbert traversal path for grid of size (2^bits × 2^bits)."""
    N = 1 << bits
    xs, ys = np.meshgrid(np.arange(N, dtype=np.uint64),
                         np.arange(N, dtype=np.uint64), indexing='xy')

    D = hilbert_xy_to_index(xs, ys, bits)
    flat = D.ravel().astype(np.uint64)
    order = np.argsort(flat)

    xy = np.column_stack((xs.ravel()[order], ys.ravel()[order])).astype(np.float64)
    path_x, path_y = xy[:, 0] + 0.5, xy[:, 1] + 0.5  # centers

    # --- Plot ---
    plt.figure(figsize=(6, 6))
    plt.plot(path_x, path_y, linewidth=1)
    plt.scatter(path_x[0], path_y[0], s=40, c='green', label='start (d=0)')
    plt.scatter(path_x[-1], path_y[-1], s=40, c='red', label=f'end (d={N*N-1})')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0, N)
    plt.ylim(0, N)
    plt.title(f"Hilbert Curve Traversal Path ({N}×{N}, bits={bits})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, which='both', linewidth=0.3, alpha=0.5)
    plt.legend()
    plt.show()


# --- Example usage ---
# plot_hilbert(bits=4)   # 16×16
# plot_hilbert(bits=5) # 32×32

def main():
    parser = argparse.ArgumentParser(description="Visualize Hilbert curve for given bit depth.")
    parser.add_argument("--bits", type=int, default=4, help="Number of bits for Hilbert curve (default: 4)")
    args = parser.parse_args()
    
    bits = args.bits
    print(f"Visualizing Hilbert curve with {bits} bits.")
    plot_hilbert(bits)
    # Call your visualization function here, e.g.:
    # visualise_hilbert_curve(bits)

if __name__ == "__main__":
    main()