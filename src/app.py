import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report

# Funcion para remover outliers
def remover_outliers(nombre_columna, nombre_dataframe,umbral = 1.5):
    """
    Funcion que calcula el rango intercuartilico (IQR)
    y elimina outliers que superan la distancia umbral*IQR:
    - para valores atípicos umbral = 1.5
    - para valores extremos umbral = 3
    Inputs:
    nombre_columna: str con nombre de la columna en la que remover outliers
    nombre_dataframe (default = df): nombre del dataframe de trabajo
    umbral (default = 1.5)
    """
    # IQR
    Q1 = np.percentile(nombre_dataframe[nombre_columna], 25,
                       interpolation = 'midpoint')
    Q3 = np.percentile(nombre_dataframe[nombre_columna], 75,
                       interpolation = 'midpoint')
    IQR = Q3 - Q1
    print("Dimensiones viejas: ", nombre_dataframe.shape)
    # Upper bound
    upper = np.where(nombre_dataframe[nombre_columna] >= (Q3+1.5*IQR))
    # Lower bound
    lower = np.where(nombre_dataframe[nombre_columna] <= (Q1-1.5*IQR))
    ''' Removing the Outliers '''
    nombre_dataframe = nombre_dataframe.drop(upper[0])
    nombre_dataframe = nombre_dataframe.drop(lower[0]).reset_index(drop = True)
    print("Nuevas dimensiones: ", nombre_dataframe.shape)
    return nombre_dataframe

# Cargo datos
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv')

# Codifico variables categoricas
sex_dic = {'male':1,'female':0}
df_raw['sex'] = df_raw['sex'].map(sex_dic)

smoker_dic = {'yes':1,'no':0}
df_raw['smoker'] = df_raw['smoker'].map(smoker_dic)

region_dic = {'southeast':0, 'southwest':1, 'northwest':2, 'northeast':3}
df_raw['region'] = df_raw['region'].map(region_dic)

# Remuevo outliers
remover_outliers('bmi', df_raw,umbral = 1.5)
remover_outliers('charges', df_raw,umbral = 1.5)

# Defino variables
X = df_raw.drop(['charges'], axis = 1)
y = df_raw.charges

# Split the dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = 0)

# Construyo el modelo
model = LinearRegression()
model.fit(X_train,y_train)

# Guardar el modelo
filename = '../models/modelo_regresion_lineal.sav'
pickle.dump(model, open(filename, 'wb'))

# Evaluo el modelo
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
RMSE_train = mean_squared_error(y_train, y_train_pred, squared=False)
RMSE_test = mean_squared_error(y_test, y_test_pred, squared=False)

print(f'The R2 score is : {r2_score(y_test,y_test_pred)}')
print('train:', RMSE_train, 'test:', RMSE_test)

