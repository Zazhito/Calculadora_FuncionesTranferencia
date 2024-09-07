import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo Excel
data = pd.read_excel('Datos_PTC.xlsx')

# Extraer las columnas de interés
tiempo = data['TIEMPO'].values
temperatura = data['TEMPERATURA'].values

# Definir la función de transferencia de segundo orden
def segundo_orden(t, K, wn, zeta):
    wd = wn * np.sqrt(1 - zeta**2)
    respuesta = K * (1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + (zeta/np.sqrt(1-zeta**2)) * np.sin(wd * t)))
    return respuesta

# Valores iniciales basados en tus cálculos previos
p0 = [1, 1, 0.1]

# Ajustar la función de transferencia a los datos usando los valores iniciales
popt, pcov = curve_fit(segundo_orden, tiempo, temperatura, p0=p0)

# Obtener los valores ajustados para K, wn, y zeta
K, wn, zeta = popt

# Calcular las incertidumbres (desviaciones estándar)
uncertainties = np.sqrt(np.diag(pcov))

# Imprimir los valores ajustados y las incertidumbres
print(f"{K}, {wn}, {zeta}")
print(f"Valores ajustados: K = {K} ± {uncertainties[0]}")
print(f"wn = {wn} ± {uncertainties[1]}")
print(f"zeta = {zeta} ± {uncertainties[2]}")

# Graficar los resultados
plt.figure()
plt.plot(tiempo, temperatura, 'b-', label='Datos')
plt.plot(tiempo, segundo_orden(tiempo, *popt), 'r--', label='Ajuste')
plt.xlabel('Tiempo')
plt.ylabel('Temperatura')
plt.legend()

# Mostrar los parámetros ajustados en la gráfica
plt.text(0.05 * max(tiempo), 0.95 * max(temperatura), 
         f"K = {K:.4f}\nwn = {wn:.4f}\nzeta = {zeta:.4f}", 
         bbox=dict(facecolor='white', alpha=0.5))

# Mostrar la gráfica
plt.show()
