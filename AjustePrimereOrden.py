import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Definir la función de transferencia de primer orden con retardo
def func_orden1_con_retrado(t, K, tau, T0):
    theta = 182  # retardo en seg
    return np.where(t >= theta, T0 + K * (1 - np.exp(-(t - theta)/tau)), T0)

# Cargar los datos desde el archivo de texto
data = np.loadtxt('Datos_PI.txt', skiprows=1)  # Ignorar la primera fila si es el encabezado
temperatura = data

# Generar un vector de tiempo para ajustarse a los datos
t = np.arange(len(temperatura))

# Valor inicial de la temperatura 
T0 = 25.06 

# Ajustar la curva usando curve_fit, asumiendo un retardo de 400 segundos
popt, pcov = curve_fit(lambda t, K, tau: func_orden1_con_retrado(t, K, tau, T0), t, temperatura, p0=[1, 1])

# Extraer los parámetros ajustados
K_ajustado, tau_ajustado = popt

# Generar los valores ajustados de temperatura
temperatura_ajustada = func_orden1_con_retrado(t, K_ajustado, tau_ajustado, T0)

# Graficar los datos originales y la curva ajustada
plt.plot(t, temperatura, 'b-', label='Datos originales')
plt.plot(t, temperatura_ajustada, 'r--', label='Curva ajustada')
plt.xlabel('Tiempo')
plt.ylabel('Temperatura')
plt.grid(True)
plt.legend()

# Mostrar los parámetros ajustados en la gráfica
plt.text(0.05 * max(t), 0.95 * max(temperatura), 
         f"K = {K_ajustado:.4f}\nTau = {tau_ajustado:.4f}", 
         bbox=dict(facecolor='white', alpha=0.5))

# Mostrar la gráfica
plt.show()

# Imprimir los parámetros ajustados
print(f"Parámetros ajustados: K = {K_ajustado}, Tau = {tau_ajustado}")