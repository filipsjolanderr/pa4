import numpy as np
import matplotlib.pyplot as plt

def logistic_loss(z):
    """Log loss function: log(1 + exp(-z))"""
    return np.log(1 + np.exp(-z))

def hinge_loss(z):
    """Hinge loss function: max(0, 1-z)"""
    return np.maximum(0, 1 - z)

# Create points for plotting
z = np.linspace(-3, 3, 1000)

# Calculate loss values
log_loss_values = logistic_loss(z)
hinge_loss_values = hinge_loss(z)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(z, log_loss_values, label='Logistic Loss (log(1 + exp(-z)))', linewidth=2)
plt.plot(z, hinge_loss_values, label='Hinge Loss (max(0, 1-z))', linewidth=2)

# Add labels and title
plt.xlabel('z = y * (w^T x)')
plt.ylabel('Loss')
plt.title('Comparison of Logistic Loss and Hinge Loss')
plt.grid(True, alpha=0.3)
plt.legend()

# Add vertical line at z=0
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Save the plot
plt.savefig('loss_comparison.png')
plt.close() 