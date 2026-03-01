import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

# Create scatter plot
plt.figure(figsize=(10, 6))
for species in iris.target_names:
    species_df = df[df['species'] == species]
    plt.scatter(species_df['sepal length (cm)'], 
                species_df['sepal width (cm)'], 
                label=species, s=100, alpha=0.7)

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Flowers - Sepal Measurements')
plt.legend()
plt.show()
