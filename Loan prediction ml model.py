import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('loan-predictionUC.csv.xlsx')

# Basic statistics
print(df.describe())

# Distribution of Loan Status
sns.countplot(data=df, x='Loan_status')
plt.title('Distribution of Loan Status')
plt.show()

# Income vs Loan Amount
sns.scatterplot(data=df, x='ApplicantIncome', y='LoanAmount', hue='Loan_status')
plt.title('Applicant Income vs Loan Amount')
plt.show()

# Property Area vs Loan Status
sns.countplot(data=df, x='Property_area', hue='Loan_status')
plt.title('Loan Status by Property Area')
plt.show()
