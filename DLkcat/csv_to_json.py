import pandas as pd

# Change dataset_name variable, the unit (line 19) and the variable name in line 31

dataset_name = 'kcat_all'


df = pd.read_csv(f"{dataset_name}.csv")

i = df[df['CMPD_SMILES'].astype(str).str.startswith('Status')].index
j = df[df['CMPD_SMILES'].astype(str).str.contains("\\\\")].index

df = df.drop(j)
l = df[df['CMPD_SMILES'].astype(str).str.startswith("[Fe+5]")].index
df = df.drop(l)
z = df[df['CMPD_SMILES'].astype(str).str.startswith("[Fe+6]")].index
df = df.drop(z)

df["Unit"] = "s^(-1)"

with open(f"{dataset_name}.json", "w") as f:
    f.write('[\n')
    for index, row in df.iterrows():
        f.write('\t{\n')

        #f.write('\t\t"ECNumber": "{}",\n'.format(row['ECNumber']))
        #f.write('\t\t"Organism": "{}",\n'.format(row['Organism']))
        f.write('\t\t"Smiles": "{}",\n'.format(row['CMPD_SMILES']))
        #f.write('\t\t"Substrate": "{}",\n'.format(row['Substrate']))
        f.write('\t\t"Sequence": "{}",\n'.format(row['SEQ']))
        f.write('\t\t"Value": "{}",\n'.format(row['kcat']))
        f.write('\t\t"Unit": "{}"\n'.format(row['Unit']))

        if index==len(df)-1:
            print("end")
            f.write('\t}\n')
            break
        else:
            f.write('\t},\n')
    f.write(']')

#C:/Users/Rana/OneDrive/Documents/GitHub/Kinetic-Parameter-Prediction-Pipeline/