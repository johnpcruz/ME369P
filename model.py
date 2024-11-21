import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, accuracy_score
import re
import matplotlib.pyplot as plt

'''
Read in data and rename columns to understandable names.
Ensure columns are numerical and create a Type column for each samples material.
Drop columns that are not wanted for testing.
'''
data = pd.read_csv("Data.csv")
#print(data.info())
# Std	ID	Material	Heat treatment	Su	Sy	A5	Bhn	E	G	mu	Ro	pH	Desc	HV
# Units are in MPa
data.rename(columns = {"Std":"Standard","Su":"UTS","Sy":"Yield","A5":"Strain",
                        "Bhn":"Brinell","E":"Elastic Mod","G":"Shear Mod",
                        "mu":"Poissons","Ro":"Density","pH":"Pressure","HV":"Vickers"}, inplace = True)

data['Yield'] = pd.to_numeric(data['Yield'], errors = 'coerce')
data = data.dropna(subset = ['Yield']).astype({'Yield': 'int64'})


# Create column with sample's material type
mat_keywords = ["steel","iron","copper","brass","bronze","aluminum","magnesium"]
def extract_keywords(row):
    result = []
    for keyword in mat_keywords:
        if pd.notna(row['Material']) and re.search(keyword, row['Material'], re.IGNORECASE):
            result.append(keyword)
        elif pd.notna(row['Desc']) and re.search(keyword, row['Desc'], re.IGNORECASE):
            result.append(keyword)
    return ', '.join(result)

data['Type'] = data.apply(extract_keywords, axis = 1)
data = data[(data['Type'] != 'bronze, aluminum') & (data['Type'] != '')]
data = data.drop(["Standard","ID","Material","Heat treatment","Desc"], axis = 1)

'''
For training the models for properties such as hardness and strain, we convert
the material type column into binary columns for the model to read.
'''
data_encoded = pd.get_dummies(data, columns = ["Type"])

'''
Brinell test data and model
'''
bH_data = data_encoded.dropna(subset = ["Brinell"])
bH_dataX = bH_data.drop(["Brinell","Pressure","Vickers"], axis = 1)
bH_dataY = bH_data["Brinell"]


bHX_train, bHX_test, bHY_train, bHY_test = train_test_split(bH_dataX, bH_dataY, test_size = 0.2, random_state = 1)
bHrf = RandomForestRegressor(n_estimators = 150, n_jobs = -1, random_state = 1)
bHrf.fit(bHX_train, bHY_train)
bHY_pred = bHrf.predict(bHX_test)

mae = mean_absolute_error(bHY_test, bHY_pred)
mse = root_mean_squared_error(bHY_test, bHY_pred)
r2 = r2_score(bHY_test, bHY_pred)

print("Brinell Hardness Model")
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

plt.scatter(bHY_test, bHY_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Brinell Actual vs Predicted')
plt.show()

importance_scores = bHrf.feature_importances_
importance_df = pd.DataFrame({'Feature': bH_dataX.columns, 'Importance': importance_scores})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)


'''
Pressure Model
'''
p_data = data_encoded.dropna(subset = ["Pressure"])
p_dataX = p_data.drop(["Brinell","Pressure","Vickers"], axis = 1)
p_dataY = p_data["Pressure"]

pX_train, pX_test, pY_train, pY_test = train_test_split(p_dataX, p_dataY, test_size = 0.2, random_state = 1)
prf = RandomForestRegressor(n_estimators = 150, n_jobs = -1, random_state = 1)
prf.fit(pX_train, pY_train)
pY_pred = prf.predict(pX_test)

mae = mean_absolute_error(pY_test, pY_pred)
mse = root_mean_squared_error(pY_test, pY_pred)
r2 = r2_score(pY_test, pY_pred)

print("\nPressure at Yield Model")
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

plt.scatter(pY_test, pY_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Pressure Actual vs Predicted')
plt.show()

importance_scores = prf.feature_importances_
importance_df = pd.DataFrame({'Feature': p_dataX.columns, 'Importance': importance_scores})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)


'''
Strain Model
'''
s_data = data_encoded.dropna(subset = ["Strain"])
s_dataX = s_data.drop(["Brinell","Pressure","Vickers","Strain"], axis = 1)
s_dataY = s_data["Strain"]

sX_train, sX_test, sY_train, sY_test = train_test_split(s_dataX, s_dataY, test_size = 0.2, random_state = 1)
srf = RandomForestRegressor(n_estimators = 150, n_jobs = -1, random_state = 1)
srf.fit(sX_train, sY_train)
sY_pred = srf.predict(sX_test)

mae = mean_absolute_error(sY_test, sY_pred)
mse = root_mean_squared_error(sY_test, sY_pred)
r2 = r2_score(sY_test, sY_pred)

print("\nStrain Model")
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

plt.scatter(sY_test, sY_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Strain Actual vs Predicted')
plt.show()

importance_scores = srf.feature_importances_
importance_df = pd.DataFrame({'Feature': s_dataX.columns, 'Importance': importance_scores})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

'''
Material Classifier
'''
matX = data.drop(["Brinell","Pressure","Vickers","Type"], axis = 1)
matY = data['Type']

matX_train, matX_test, matY_train, matY_test = train_test_split(matX, matY, test_size=0.2, random_state=1)

mat_classifier = RandomForestClassifier(n_estimators = 150, n_jobs = -1, random_state = 1)
mat_classifier.fit(matX_train, matY_train)

mat_pred = mat_classifier.predict(matX_test)

accuracy = accuracy_score(matY_test, mat_pred)
print("\nMaterial Classifier Model")
print(f"Accuracy: {accuracy}")

importance_scores = mat_classifier.feature_importances_
importance_df = pd.DataFrame({'Feature': matX.columns, 'Importance': importance_scores})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

#new_sample = pd.DataFrame({'UTS':[552],'Yield':[186],'Strain':[35],
#                     'Elastic Mod':[120000],'Shear Mod':[45000],'Poissons':[0.32],
#                     'Density':[8800]})
# new_sample = pd.DataFrame({'UTS':[950],'Yield':[750],'Strain':[None],
#                      'Elastic Mod':[211000],'Shear Mod':[82000],'Poissons':[0.29],
#                      'Density':[7640]})
# predicted_material = mat_classifier.predict(new_sample)
# print(f"Predicted Material Type: {predicted_material}")

def classifier(data):
    prediction = mat_classifier.predict(data)
    return prediction


'''
Material Verifier
'''
def verifier(data,classifier=mat_classifier):
    results = []
    for _, row in data.iterrows():
        features = row.drop(['Type']).values.reshape(1, -1)
        
        predicted_type = classifier.predict(features)[0]
        
        confidence = max(classifier.predict_proba(features)[0])
        
        actual_type = row['Type']
        match = (predicted_type == actual_type) if pd.notna(actual_type) else None
        
        # Append results
        results.append({
            'Actual_Type': actual_type,
            'Predicted_Type': predicted_type,
            'Match': match,
            'Confidence': confidence
        })
    
    # Return results as a DataFrame
    return pd.DataFrame(results)

#test_data = pd.DataFrame({'UTS':[150],'Yield':[120],'Strain':[6],
#                     'Elastic Mod':[45000],'Shear Mod':[18000],'Poissons':[0.35],
#                     'Density':[1740],'Type':['magnesium']})
# test_data = pd.DataFrame({'UTS':[850],'Yield':[630],'Strain':[13],
#                      'Elastic Mod':[206000],'Shear Mod':[80000],'Poissons':[0.3],
#                      'Density':[7860],'Type':['steel']})
# verification_results = verifier(test_data, mat_classifier)
# print(verification_results)




'''
Get mean values for columns depending on material type
'''
mean_values = data.groupby('Type').mean()
mean_values = mean_values.drop(['Brinell','Pressure','Vickers'], axis = 1)
#print(mean_values)

def imputer(row, mean_values, target, classifier):
    if pd.isna(row['Type']):
        features = row.drop(['Type']).values.reshape(1, -1)
        row['Type'] = classifier.predict(features)[0]  # Predict Type
    
    
    material_type = row['Type']
    for column in mean_values.columns:
        if pd.isna(row[column]) and column != target:
            row[column] = mean_values.loc[material_type, column]
    return row

'''
Uses imputer function to fill missing values if needed, and makes sure 
the one-hot encoding process done at the beginning is taken into account.
'''
def predict(model, data, mean_values, target, classifier = mat_classifier):
    if model == srf:
        dataX = s_dataX
    else:
        dataX = bH_dataX
    imputed = data.apply(lambda row: imputer(row, mean_values, target, classifier), axis = 1)
    imputed_encoded = pd.get_dummies(imputed, columns = ['Type'])
    expected_columns = dataX.columns
    for col in expected_columns:
        if col not in imputed_encoded:
            imputed_encoded[col] = 0
    imputed_encoded = imputed_encoded[expected_columns]
    
    prediction = model.predict(imputed_encoded)
    
    return prediction

# test = pd.DataFrame({'UTS':[386],'Yield':[284],'Strain':[37],
#                      'Elastic Mod':[None],'Shear Mod':[None],'Poissons':[None],
#                      'Density':[7860],'Type':'steel'})
# test1 = pd.DataFrame({'UTS':[440],'Yield':[370],'Strain':[None],
#                      'Elastic Mod':[205000],'Shear Mod':[80000],'Poissons':[0.29],
#                      'Density':[7870],'Type':[None]})

# prediction = predict(bHrf, test1, mean_values, target = "Brinell")
# print(prediction)


# test2 = pd.DataFrame({'UTS':[900],'Yield':[680],'Strain':[10],
#                      'Elastic Mod':[206000],'Shear Mod':[80000],'Poissons':[0.3],
#                      'Density':[7860],'Type':['steel']})

# test3 = pd.DataFrame({'UTS':[850],'Yield':[630],'Strain':[13],
#                      'Elastic Mod':[206000],'Shear Mod':[80000],'Poissons':[0.3],
#                      'Density':[7860],'Type':['steel']})

# test4 = pd.DataFrame({'UTS':[421],'Yield':[314],'Strain':[None],
#                      'Elastic Mod':[207000],'Shear Mod':[79000],'Poissons':[0.3],
#                      'Density':[7860],'Type':['steel']})
# prediction = predict(vrf, test4, mean_values, target = "Vickers")
# print(prediction)
# prediction = predict(srf, test4, mean_values, target = "Strain")
# print(prediction)
 
def all_predictions(data, models, classifier=mat_classifier):
    
    results = {}
    for target, model in models.items():
        prediction = predict(model, test_data.copy(), mean_values, target, classifier)
        results[target]=prediction[0]
    results = pd.DataFrame([results])
    return results

def material_input():
    print("Please enter the following values for the material (leave blank if unknown):")
    uts = input("Ultimate Tensile Strength (UTS) (MPa): ")
    yield_strength = input("Yield Strength (MPa): ")
    strain = input("Strain (%): ")
    elastic_mod = input("Elastic Modulus (MPa): ")
    shear_mod = input("Shear Modulus (MPa): ")
    poissons = input("Poisson's Ratio: ")
    density = input("Density (kg/m^3): ")
    material_type = input("Material Type (e.g., steel, brass, etc.): ")
    
    data = {
        'UTS': [float(uts) if uts else None],
        'Yield': [float(yield_strength) if yield_strength else None],
        'Strain': [float(strain) if strain else None],
        'Elastic Mod': [float(elastic_mod) if elastic_mod else None],
        'Shear Mod': [float(shear_mod) if shear_mod else None],
        'Poissons': [float(poissons) if poissons else None],
        'Density': [float(density) if density else None],
        'Type': [material_type if material_type else None]
    }

    return pd.DataFrame(data)

import warnings
warnings.simplefilter("ignore", UserWarning)

if __name__ == '__main__':
    
    
    testcase = input("\nEnter 1 to predict material properties, 2 to predict material types, 3 to verify material, or q to quit: ")
    while testcase != 'q':
        if testcase == '1':
            test_data = material_input()
            print("Inputed Material: ")
            print(test_data)
            if pd.notna(test_data['Strain'].iloc[0]):
                models = {"Brinell":bHrf,"Pressure":prf}
            else:
                models = {"Brinell":bHrf,"Pressure":prf,"Strain":srf}
            predictions = all_predictions(test_data, models)
            print("Predicted Material Properties")
            print(predictions)
        elif testcase == '2':
            test_data = material_input()
            test_data = test_data.drop(['Type'], axis=1)
            print("Inputed Material: ")
            print(test_data)
            predicted_material = classifier(test_data)
            print(f"Predicted Material Type: {predicted_material}")
        elif testcase == '3':
            test_data = material_input()
            comparison = verifier(test_data)
            print(comparison)
        elif testcase == 'q':
            break
        else:
            print("Invalid. Try Again")
        testcase = input("\nEnter 1 to predict material properties, 2 to predict material types, 3 to verify material, or q to quit: ")            
    
    

    