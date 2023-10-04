import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style = 'darkgrid')



file_path = r'C:\Users\PC\OneDrive\Escritorio\DS_Bootcamp\CURSO\28_ML_WEBAPP_USING_STREAMLIT\julio-segura-streamlit-tutorial-main\data\raw\Salary.csv'
dataframe = pd.read_csv(file_path)


from sklearn.preprocessing import MinMaxScaler
num_variables = ['YearsExperience', 'Salary']
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(dataframe[num_variables])
total_data_scal = pd.DataFrame(scal_features, index = dataframe.index, columns = num_variables)
total_data_scal.head()


x = dataframe.iloc[:,:1].values
y = dataframe.iloc[:,1:]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


from sklearn.linear_model import LinearRegression


model = LinearRegression()


model.fit(x_train, y_train)


y_pred = model.predict(x_test)


from pickle import dump

dump(model, open("linear_regression_model_42.sav", "wb"))






