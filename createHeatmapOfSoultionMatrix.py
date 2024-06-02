import requests
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

basepath = "http://85.215.235.161:8080/getAllSolutions?levelId="

result = requests.get(basepath + "4")

json = result.json()
solution = json[0]["solutionMatrix"]



sns.color_palette("rocket", as_cmap=True)
sns.heatmap(solution)
plt.title('Heatmap for level "Howaka 4"')
plt.show()