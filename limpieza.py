import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'data\raw\data_inmobiliario.csv', encoding='latin1')### utf-16


sns.set(style="whitegrid")
df.columns
df.dtypes


print(df.head())
print(df.info())
print(df.describe())
print(df.describe(include='object'))
print(df.isnull().sum())

# Histograma variables numéricas
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()


### columnas catgoricas
categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, data=df)
    plt.title(f'Distribución de {col}')
    plt.xticks(rotation=45)
    plt.show()

### Matriz correlación

df_numeric = df.select_dtypes(include=['float64', 'int64'])

corr_matrix = df_numeric.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')

plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.title('Matriz de Correlación')
plt.tight_layout()
plt.show()
