import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load data
ages = pd.read_csv('Ages.csv')
data = pd.read_csv('data.csv')

data.rename(columns={'Unnamed: 0': 'Sample Accession'}, inplace=True)
data_cleaned = data.drop(data.columns[1], axis=1)
merged_data = pd.merge(ages, data_cleaned, on='Sample Accession')

X = merged_data.drop(columns=['Sample Accession', 'Age'])
y = merged_data['Age']

X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0) 

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# GUI setup
root = tk.Tk()
root.title("BS429 Introduction To Pattern Recognition Report")
root.geometry("800x600")

frame_left = ttk.Frame(root, padding="10", height=200, width=400)
frame_left.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

frame_right = ttk.Frame(root, padding="10", height=200, width=400)
frame_right.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

style = ttk.Style()
style.configure("Green.TFrame", background="#b8e5b8")
style.configure("Pink.TFrame", background="#f2b5d4")

frame_left.configure(style="Green.TFrame")
frame_right.configure(style="Pink.TFrame")

ttk.Label(frame_left, text="MAE", font=("Helvetica", 14, "bold"), background="#b8e5b8").grid(column=1, row=1)
ttk.Label(frame_right, text="R^2", font=("Helvetica", 14, "bold"), background="#f2b5d4").grid(column=1, row=1)

ttk.Label(frame_left, text=f"{mae:.2f}", font=("Helvetica", 16), background="#b8e5b8").grid(column=1, row=2, sticky=tk.W)
ttk.Label(frame_right, text=f"{r2:.2f}", font=("Helvetica", 16), background="#f2b5d4").grid(column=1, row=2, sticky=tk.W)

# Scatter plot
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(y_test, y_pred, c='blue', label='Predicted vs Actual')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
ax.set_xlabel('Actual Ages')
ax.set_ylabel('Predicted Ages')
ax.set_title('Predicted vs Actual Ages')
ax.legend()

# Embed the plot into the GUI
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=2, column=0, columnspan=2, pady=20)
canvas.draw()

root.mainloop()