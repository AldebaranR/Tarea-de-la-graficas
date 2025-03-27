import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

try:
    df = pd.read_excel("proyecto1.xlsx")  
except Exception as e:
    print("Error al leer el archivo:", e)
    sys.exit()

print("Columnas en el DataFrame:", df.columns)

if 'B_mes' in df.columns:
    df['B_mes'] = pd.to_datetime(df['B_mes'], dayfirst=True, errors='coerce')

ventas_totales = df['ventas_tot'].sum()
print("Ventas totales del comercio:", ventas_totales)

df_con_adeudo = df[df['B_adeudo'].str.strip().str.lower() == "con adeudo"]
df_sin_adeudo = df[df['B_adeudo'].str.strip().str.lower() == "sin adeudo"]

socios_con_deuda = df_con_adeudo['no_clientes'].sum()
socios_sin_deuda = df_sin_adeudo['no_clientes'].sum()
total_socios = df['no_clientes'].sum()

porcentaje_con_deuda = (socios_con_deuda / total_socios) * 100 if total_socios else 0
porcentaje_sin_deuda = (socios_sin_deuda / total_socios) * 100 if total_socios else 0

print(f"Socios (clientes) con deuda: {socios_con_deuda} ({porcentaje_con_deuda:.2f}%)")
print(f"Socios (clientes) sin deuda: {socios_sin_deuda} ({porcentaje_sin_deuda:.2f}%)")

# Gráfica 1: Ventas totales respecto al tiempo
if 'B_mes' in df.columns and df['B_mes'].notnull().any():
    ventas_por_bmes = df.groupby(df['B_mes'].dt.date)['ventas_tot'].sum()

    plt.figure(figsize=(10, 6))
    ventas_por_bmes.plot(kind='bar', color='purple')  # Nuevo color
    plt.title("Ventas Totales Respecto del Tiempo (B_mes)")
    plt.xlabel("B_mes")
    plt.ylabel("Ventas Totales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No hay datos de B_mes para graficar ventas totales respecto del tiempo.")

# Gráfica 2: Desviación estándar de los pagos
if 'B_mes' in df.columns and df['B_mes'].notnull().any():
    std_pagos_por_bmes = df.groupby(df['B_mes'].dt.date)['pagos_tot'].std()

    plt.figure(figsize=(10, 6))
    std_pagos_por_bmes.plot(kind='bar', color='black')  # Nuevo color
    plt.title("Desviación Estándar de los Pagos Totales por B_mes")
    plt.xlabel("B_mes")
    plt.ylabel("Desviación Estándar de Pagos")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No hay datos de B_mes para graficar la desviación estándar de los pagos.")

deuda_total = df['adeudo_actual'].sum()
print("Deuda total de los clientes:", deuda_total)

if ventas_totales != 0:
    porcentaje_utilidad = ((ventas_totales - deuda_total) / ventas_totales) * 100
else:
    porcentaje_utilidad = 0
print(f"Porcentaje de utilidad del comercio: {porcentaje_utilidad:.2f}%")

# Gráfica 3: Ventas por sucursal (pastel)
ventas_por_sucursal = df.groupby('id_sucursal')['ventas_tot'].sum()

plt.figure(figsize=(8, 8))
ventas_por_sucursal.plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=90,
    colors=['lightcoral', 'cornflowerblue', 'mediumorchid', 'lightgreen', 'orange']  # Nuevos colores
)
plt.title("Ventas por Sucursal")
plt.ylabel("")  # Ocultar etiqueta del eje Y
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 8) Gráfico de barras: Deuda total y Margen de utilidad por sucursal
# -------------------------------------------------------

datos_sucursal = df.groupby('id_sucursal').agg({
    'adeudo_actual': 'sum',
    'ventas_tot': 'sum'
}).reset_index()

# Función para calcular el margen de utilidad
def calc_margen_utilidad(row):
    if row['ventas_tot'] != 0:
        return (row['ventas_tot'] - row['adeudo_actual']) / row['ventas_tot'] * 100
    else:
        return 0

datos_sucursal['margen_utilidad'] = datos_sucursal.apply(calc_margen_utilidad, axis=1)

# Gráfico de barras con dos ejes Y
x = np.arange(len(datos_sucursal))
width = 0.4

fig, ax1 = plt.subplots(figsize=(12, 6))

# Primer eje Y - Deuda total
bars1 = ax1.bar(x - width/2, datos_sucursal['adeudo_actual'], width, label='Deuda Total', color='indianred')
ax1.set_xlabel('ID Sucursal')
ax1.set_ylabel('Deuda Total', color='indianred')
ax1.tick_params(axis='y', labelcolor='indianred')

# Segundo eje Y - Margen de utilidad
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, datos_sucursal['margen_utilidad'], width, label='Margen de Utilidad (%)', color='steelblue')
ax2.set_ylabel('Margen de Utilidad (%)', color='steelblue')
ax2.tick_params(axis='y', labelcolor='steelblue')

# Etiquetas en eje X
plt.xticks(x, datos_sucursal['id_sucursal'])
plt.title('Deuda Total y Margen de Utilidad por Sucursal')

# Combinar leyendas de ambos ejes
lines_labels_1 = ax1.get_legend_handles_labels()
lines_labels_2 = ax2.get_legend_handles_labels()
lines = lines_labels_1[0] + lines_labels_2[0]
labels = lines_labels_1[1] + lines_labels_2[1]
fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.1, 0.9))

plt.tight_layout()
plt.show()