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