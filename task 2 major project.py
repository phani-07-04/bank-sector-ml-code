import pandas as pd
from tabulate import tabulate

# Load the dataset
df = pd.read_csv('loan-predictionUC.csv.xlsx')

# Display basic statistics in a table
print("Basic Statistics:")
print(tabulate(df.describe(), headers='keys', tablefmt='pretty'))

# Display the first few rows of the dataset in a table
print("\nFirst Few Rows of the Dataset:")
print(tabulate(df.head(), headers='keys', tablefmt='pretty'))
