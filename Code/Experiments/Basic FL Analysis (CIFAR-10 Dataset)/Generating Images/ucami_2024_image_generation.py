import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("cifar-10_epochs_rounds_data.csv")
data = pd.DataFrame(data)

# print(data.head())
client_num = data["Clients"] 
rounds = data.columns
rounds = rounds[1:]
x_plot = data["Clients"]
y_plot = data.drop("Clients", axis=1)

# Iterate over the columns in y_plot to plot each as a separate line
for column in y_plot.columns:
    plt.plot(x_plot, y_plot[column], label=column)

plt.legend(title="Training Rounds", loc='upper right',fontsize='small',title_fontsize='small', ncol=2)  # This will add the legend to the plot
# plt.legend(title="Training Rounds", loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', title_fontsize='medium')
plt.xlabel("Clients")
plt.ylabel("Accuracy (%)")
plt.ylim([0,100])
plt.xlim([0,100])
plt.grid(True) # Add grid lines to the plot
plt.title("Accuracy Results For Model Training with Various \nClient Numbers and Various Training Rounds")
plt.show()