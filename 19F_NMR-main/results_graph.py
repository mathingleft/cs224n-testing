import matplotlib.pyplot as plt

# Sample data
models = ['MLP', 'SVR', 'RF', 'DT', 'GBR', 'GPR', 'KNN', 'EN', 'BR']
mae_values = [22.483, 21.69, 16.215,22.055, 20.242,22.33,20.283, 22.359, 21.687]  # Mean Absolute Error values
r2_values = [-0.009, 0.0238, 0.314,0.0136, 0.1599,0.009,0.102, 0.00840, 0.0240]  # R^2 values

fig, ax1 = plt.subplots()

# Create bar chart
ax1.bar(models, mae_values, color='skyblue')

# Add labels for MAE
for i in range(len(models)):
    ax1.text(models[i], mae_values[i] + 0.5, f"MAE: {mae_values[i]}", ha='center')

# Add secondary y-axis for R^2
ax2 = ax1.twinx()
ax2.plot(models, r2_values, marker='o', linestyle='-', color='orange')

# Add labels for R^2
for i in range(len(models)):
    ax2.text(models[i], r2_values[i] + 0.02, f"R$^2$: {r2_values[i]}", ha='center')

# Add title and labels
ax1.set_title('Mean Absolute Error and R$^2$ Comparison')
ax1.set_xlabel('Models')
ax1.set_ylabel('Mean Absolute Error')
ax2.set_ylabel('R$^2$')

# Show plot
plt.grid(True)
plt.show()