import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import os

def plot_corr_matrix(corr, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    corr_real = np.real(corr)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_real, cmap='coolwarm', center=0, square=True)
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(output_dir, "corr_matrix.png"))
    plt.close()

def plot_cgc_timeseries(cgc, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    times = cgc.coords['times'].values
    for i, roi in enumerate(cgc.coords['roi'].values):
        plt.figure(figsize=(10, 6))
        for j, direction in enumerate(cgc.coords['direction'].values):
            plt.plot(times, cgc[i, :, j], label=str(direction))
        plt.xlabel("Time (s)")
        plt.ylabel("cGC")
        plt.title(f"cGC for ROI: {roi}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"cgc_{roi}.png"))
        plt.close()

def plot_freq_response(data, title, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    num_channels = data.shape[0]
    for i in range(num_channels):
        plt.figure(figsize=(10, 5))
        for j in range(num_channels):
            plt.plot(data[i, j, :], label=f"{i}->{j}")
        plt.xlabel("Frequency bin")
        plt.ylabel("Value")
        plt.title(f"{title}: From channel {i}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{title.lower()}_from_{i}.png"))
        plt.close()

def visualize_all_results(results, output_dir="results"):
    print("Generating visualizations...")
    if 'corr' in results:
        plot_corr_matrix(results['corr'], output_dir)
    if 'cgc' in results:
        plot_cgc_timeseries(results['cgc'], output_dir)
    if 'adtf' in results:
        plot_freq_response(results['adtf'], "ADTF", output_dir)
    if 'dtf' in results:
        plot_freq_response(results['dtf'], "DTF", output_dir)
    if 'pdcoh' in results:
        plot_freq_response(results['pdcoh'], "PDCoh", output_dir)
    print(f"All plots saved in '{output_dir}'.")


