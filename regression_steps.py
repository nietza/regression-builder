import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy import stats

class RegressionSteps(tk.Toplevel):
    def __init__(self, parent, x_data, y_data):
        super().__init__(parent)
        
        self.title("Regression Analysis Steps")
        self.geometry("800x600")
        
        self.x_data = x_data
        self.y_data = y_data
        
        self.setup_ui()
        self.calculate_steps()
        
    def setup_ui(self):
        self.text_frame = ttk.Frame(self, padding=20)
        self.text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.text_widget = tk.Text(
            self.text_frame,
            wrap=tk.WORD,
            font=('Consolas', 11),
            bg='#2b3e50',
            fg='white'
        )
        
        scrollbar = ttk.Scrollbar(
            self.text_frame,
            orient="vertical",
            command=self.text_widget.yview
        )
        self.text_widget.configure(yscrollcommand=scrollbar.set)
        
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        ttk.Button(
            self,
            text="Close",
            command=self.destroy,
            width=15
        ).pack(pady=20)
        
    def calculate_steps(self):
        n = len(self.x_data)
        x_mean = np.mean(self.x_data)
        y_mean = np.mean(self.y_data)
        
        sum_xy = np.sum(self.x_data * self.y_data)
        sum_x2 = np.sum(self.x_data ** 2)
        sum_y2 = np.sum(self.y_data ** 2)
        
        # Calculate slope and intercept
        slope = (sum_xy - n * x_mean * y_mean) / (sum_x2 - n * x_mean ** 2)
        intercept = y_mean - slope * x_mean
        
        # Calculate regression line
        y_pred = slope * self.x_data + intercept
        
        # Calculate R-squared
        ss_res = np.sum((self.y_data - y_pred) ** 2)
        ss_tot = np.sum((self.y_data - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate standard errors
        se_slope = np.sqrt(ss_res / (n - 2) / (sum_x2 - n * x_mean ** 2))
        se_intercept = np.sqrt(ss_res / (n - 2) * (1/n + x_mean**2 / (sum_x2 - n * x_mean**2)))
        
        # Calculate t-statistics and p-values
        t_slope = slope / se_slope
        t_intercept = intercept / se_intercept
        p_slope = 2 * (1 - stats.t.cdf(abs(t_slope), n - 2))
        p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), n - 2))
        
        steps_text = f"""
Step-by-Step Linear Regression Analysis
=====================================

1. Data Summary:
---------------
• Number of data points (n): {n}
• Mean of X (x̄): {x_mean:.4f}
• Mean of Y (ȳ): {y_mean:.4f}

2. Calculating Sums:
------------------
• Σ(xy): {sum_xy:.4f}
• Σ(x²): {sum_x2:.4f} 
• Σ(y²): {sum_y2:.4f}

3. Regression Coefficients:
-------------------------
• Slope (β₁) = (Σ(xy) - n·x̄·ȳ) / (Σ(x²) - n·x̄²)
• Slope = {slope:.4f}
• Standard Error of Slope = {se_slope:.4f}
• t-statistic for Slope = {t_slope:.4f}
• p-value for Slope = {p_slope:.4f}

• Intercept (β₀) = ȳ - β₁·x̄
• Intercept = {intercept:.4f} 
• Standard Error of Intercept = {se_intercept:.4f}
• t-statistic for Intercept = {t_intercept:.4f}
• p-value for Intercept = {p_intercept:.4f}

4. Regression Equation:
---------------------
y = {slope:.4f}x + {intercept:.4f}

5. Coefficient of Determination (R²):
----------------------------------
R² = {r_squared:.4f}

This means that {(r_squared * 100):.1f}% of the variance in Y 
can be explained by the linear relationship with X.

6. Interpretation:
----------------
• For each unit increase in X, Y changes by {slope:.4f} units on average
• When X = 0, Y is predicted to be {intercept:.4f} on average
• The model explains {(r_squared * 100):.1f}% of the variability in the data
"""
        
        self.text_widget.insert('1.0', steps_text)
        self.text_widget.configure(state='disabled')