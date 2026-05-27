# Note: This visualization script was generated with AI assistance to diagnose the custom ML engine.
import numpy as np
import matplotlib.pyplot as plt

class DataVisualizer:
    @staticmethod
    def plot_scaling_distributions(X_unscaled, X_scaled, feature_names=["Feature 0", "Feature 1", "Feature 2", "Feature 3"]):
        k = len(feature_names)
        fig, axes = plt.subplots(2, k, figsize=(16, 8))
        fig.suptitle("Feature Distributions: Unscaled (Top) vs. Scaled (Bottom)", fontsize=16, fontweight='bold')

        for i in range(k):
            axes[0, i].hist(X_unscaled[:, i], bins=30, density=True, histtype='step', color='blue', linewidth=2.5)
            axes[0, i].set_title(f"Raw {feature_names[i]}")
            axes[0, i].set_ylabel("Density")
            axes[0, i].grid(True, linestyle='--', alpha=0.6)

        for i in range(k):
            axes[1, i].hist(X_scaled[:, i], bins=30, density=True, histtype='step', color='red', linewidth=2.5)
            axes[1, i].set_title(f"Scaled {feature_names[i]}")
            axes[1, i].set_ylabel("Density")
            axes[1, i].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def run_and_visualize_gradient_descent(model, feature_index=0, feature_names=["Feature 0"]):
        print(f"\n--- Running Diagnostic Gradient Descent on '{feature_names[feature_index]}' ---")
        alpha = 0.01
        EPSILON = 1e-6 

        k = len(model.X[0]) 
        w = np.array([0.0] * k)
        b = 0.0

        w_hist = []
        b_hist = []

        cost = model.calculate_cost(w, b)
        prev_cost = cost + 10  

        while abs(prev_cost - cost) > EPSILON:
            w_hist.append(w[feature_index])
            b_hist.append(b)

            prev_cost = cost
            dj_dw, dj_db = model.calculate_derivatives(w, b)
            w = w - alpha * dj_dw
            b = b - alpha * dj_db
            cost = model.calculate_cost(w, b)

        w_hist.append(w[feature_index])
        b_hist.append(b)
        
        model.w = w
        model.b = b
        print("Convergence Reached! Calculating topographical terrain... (this takes a second)")

        w_hist = np.array(w_hist)
        b_hist = np.array(b_hist)

        w_margin = (np.max(w_hist) - np.min(w_hist)) * 0.5
        b_margin = (np.max(b_hist) - np.min(b_hist)) * 0.5
        w_margin = w_margin if w_margin > 0 else 10
        b_margin = b_margin if b_margin > 0 else 10

        w_vals = np.linspace(np.min(w_hist) - w_margin, np.max(w_hist) + w_margin, 50)
        b_vals = np.linspace(np.min(b_hist) - b_margin, np.max(b_hist) + b_margin, 50)
        
        W, B = np.meshgrid(w_vals, b_vals)
        J_grid = np.zeros_like(W)

        fixed_w = model.w.copy() 
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                fixed_w[feature_index] = W[i, j]
                J_grid[i, j] = model.calculate_cost(fixed_w, B[i, j])

        plt.figure(figsize=(10, 7))
        levels = np.logspace(np.log10(np.min(J_grid)), np.log10(np.max(J_grid)), 30)
        cp = plt.contour(W, B, J_grid, levels=levels, cmap='viridis', alpha=0.7)
        plt.colorbar(cp, label='Cost (J)')

        plt.plot(w_hist, b_hist, marker='o', color='red', markersize=4, linestyle='-', linewidth=2, label='Trajectory')
        plt.plot(w_hist[0], b_hist[0], marker='s', color='blue', markersize=8, label='Start')
        plt.plot(w_hist[-1], b_hist[-1], marker='*', color='gold', markeredgecolor='black', markersize=15, label='Global Minimum')

        plt.title(f"Gradient Descent Path: Weight ({feature_names[feature_index]}) vs Bias", fontsize=14, fontweight='bold')
        plt.xlabel(f"Weight for {feature_names[feature_index]}")
        plt.ylabel("Bias (b)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()