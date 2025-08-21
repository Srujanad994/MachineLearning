import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from category_encoders import TargetEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

df = pd.read_csv("Data/train.csv")
df_test = pd.read_csv("Data/test.csv")
df_sb = pd.read_csv("Data/sample_submission.csv")

pd.set_option("display.Max_columns",100)
pd.set_option("display.Max_rows",100)
df["flag"] = 0
df_test["flag"] = 1
feature = df["SalePrice"]
df.shape
df_test.shape
all = pd.concat([df,df_test], ignore_index = True)
print(all)
all.shape
all = all.drop(columns = ["Id","Utilities","SalePrice"])
all["CentralAir"] = all["CentralAir"].map({"Y": 1, "N": 0})
all.shape
for i in all.columns:
    if all[i].dtype == "object":
        all[i] = all[i].fillna(all[i].value_counts().idxmax())
    else:
        all[i] = all[i].fillna(all[i].mean())
check = []
for a in all.columns:
    if all[a].apply(type).nunique() > 1:
        lala.append(a)

# check is showing 0 values
df = all[all["flag"] == 0].drop(columns = ["flag"])
df_test = all[all["flag"] == 1].drop(columns = ["flag"])
x = df
y = feature
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.20, random_state = 18)
col = x.select_dtypes(include = ["object"]).columns

encoder = TargetEncoder(cols = col)
x_train = encoder.fit_transform(x_train,y_train)
x_test = encoder.transform(x_test)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model = LinearRegression()
model.fit(x_train,y_train)
print(model.intercept_)
print(model.coef_)
y_pred = model.predict(x_test)
print("Mean Squared Error: ",mean_squared_error(y_test,y_pred))
print("r2 Score: ", r2_score(y_test,y_pred))
print("Mean Absolute Error: ",mean_absolute_error(y_test,y_pred))
print("Mean Absolute Percentage Error: ",mean_absolute_percentage_error(y_test,y_pred))
plt.scatter(y_pred,y_test)
plt.plot(y_test,y_test, color = "red")
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.show()
print(x_train.shape)
print(df_test.shape)
df_test = encoder.transform(df_test)
df_test = scaler.transform(df_test)
y_predict = model.predict(df_test)
submission = pd.DataFrame({
    "Id": df_sb["Id"],
    "SalePrice": y_predict
})
submission.to_csv("submission.csv",index = False)