import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


df1 = pd.read_csv("Data/Cleaned Data/Cleaned_Funding_Details.csv")
df2 = pd.read_csv("Data/Cleaned Data/Cleaned_Overview_Details.csv")

data = pd.merge(df1,df2, left_on = ["Name"], right_on = ["Crypto Name"])

data['First Funding Date'] = pd.to_datetime(data['First Funding Date'])

# Extract year, month, and day components
data['First Funding Year'] = data['First Funding Date'].dt.year
data['First Funding Month'] = data['First Funding Date'].dt.month
data['First Funding Day'] = data['First Funding Date'].dt.day

data['Valuation Amount'] = data['Valuation Amount'].str.extract(r'\$ (\d+\.?\d*)M').astype(float) * 1e6

data.to_csv("Data/data.csv")
# data = pd.read_csv("Data/data.csv")

processed_data = data.drop(columns=["Crypto Name", "Name", "Raised Amount", "First Funding Date", "Valuation Amount", "Links"])

processed_data.fillna("not available", inplace=True)
encoder = OneHotEncoder()
processed_data = pd.get_dummies(processed_data, columns= ["Service", "Funding Round"], dtype = int)
processed_data["Inverstors_and_desc"] = processed_data["Investors"]+" "+processed_data["Description"]
processed_data = processed_data.drop(columns=["Investors","Description"])
# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# List of numeric column names
numeric_columns = ['Total Raised', 'First Funding Year', 'First Funding Month',
       'First Funding Day', 'Service_Blockchain Infrastructure',
       'Service_Blockchain Service', 'Service_CeFi', 'Service_Chain',
       'Service_DeFi', 'Service_GameFi', 'Service_Social',
       'Service_Stablecoin', 'Funding Round_Angel', 'Funding Round_Pre-Seed',
       'Funding Round_Pre-Series A', 'Funding Round_Seed',
       'Funding Round_Series A', 'Funding Round_Strategic',
       'Funding Round_Undisclosed']

# Apply min-max normalization to each numeric column
processed_data[numeric_columns] = scaler.fit_transform(processed_data[numeric_columns])

processed_data.to_csv("Data/processed_data.csv")