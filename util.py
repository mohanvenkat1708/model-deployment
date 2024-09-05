import pandas as pd
excel_file = pd.read_excel('./sampledata.xlsx')
excel_file.to_csv('sampledata.csv', index=False)