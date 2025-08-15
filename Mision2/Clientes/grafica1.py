import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
#=========================
# 1. Carga de datos
#===========================
df = pd.read_csv('clientes_supermercado.csv')
print(df)
#==================================
# Redondear antes de convertir a int
#===================================
df['Edad']=df['Edad'].round().astype(int)
df['VisitasPorMes']=df['VisitasPorMes'].round().astype(int)
print(df)
x=df[['Edad','GastoMensual','VisitasPorMes']]
#==========================
#Entrenar K-Means con k=4
#==========================
kmeans=KMeans(n_clusters=4,random_state=42)
kmeans.fit(x)
df['Cluster']=kmeans.labels_
#===============================
# Graficar en 3D con anotaciones
#===============================
fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(111,projection='3d')
# colocar colores a los cluster
sc =ax.scatter(df['Edad'],df['GastoMensual'],df['VisitasPorMes'],
               c=df['Cluster'],cmap='viridis',s=50)
# Etiquetas de ejes
ax.set_xlabel('Edad')
ax.set_ylabel('Gastos Mensuales')
ax.set_zlabel('Visitas por mes')
plt.title('Segmentación de clientes Kmeans k=4')
# Anotaciones interpretativas
ax.text(22,80,2,'Jóvenes bajo gasto\n Pocas visitas',color='black')
ax.text(38,480,9,'Familias gasto alto\n visitas frecuentes',color='black')
ax.text(63,310,5,'Mayores gasto medio\n visitas regulares',color='black')
ax.text(33,920,11,'Clientes VIP\n Gastos muy alto',color='black')
plt.show()
