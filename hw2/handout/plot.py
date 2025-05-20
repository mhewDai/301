import matplotlib.pyplot as plt
import numpy as np

# Replace these lists with your actual error rates
train_errors = [0.49, 0.215, 0.215, 0.125]  # placeholder values for training errors
test_errors = [0.4021, 0.2784, 0.3299, 0.2577]   # placeholder values for testing errors
max_depths = [0, 1, 2, 4]  # The depths you have evaluated

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(max_depths, train_errors, label='Training Error', marker='o')
plt.plot(max_depths, test_errors, label='Testing Error', marker='s')

plt.title('Error vs Tree Depth for Heart Disease Dataset')
plt.xlabel('Max Depth of Tree')
plt.ylabel('Error Rate')
plt.xticks(max_depths)  # This ensures only the depths used are marked
plt.legend()
plt.grid(True)
plt.show()
