# imports
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV, LassoCV


# dataset diabetes
diabetes_dataset = load_diabetes()
diabetes_data = diabetes_dataset["data"]
diabetes_target = diabetes_dataset["target"]
X_train, X_test, y_train, y_test = train_test_split(diabetes_data,
                                                    diabetes_target,
                                                    test_size=0.3,
                                                    random_state=91)

# create a model
print("LinearRegression")
lr_model = LinearRegression()

# train the model
lr_model.fit(X_train, y_train)
# print(lr_model.coef_)
# print(lr_model.intercept_)

# test the model
output = lr_model.predict(X_test)

# # The mean squared error
print("Mean squared error: %.3f" % mean_squared_error(y_test, output))
# # The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.3f" % r2_score(y_test, output))


print("ridge")
lr_model = RidgeCV()
lr_model.fit(X_train, y_train)
# print(lr_model.coef_)
# print(lr_model.intercept_)
output = lr_model.predict(X_test)
print("Mean squared error: %.3f" % mean_squared_error(y_test, output))
print("Coefficient of determination: %.3f" % r2_score(y_test, output))

print("LASSO")
lr_model = LassoCV()
lr_model.fit(X_train, y_train)
# print(lr_model.coef_)
# print(lr_model.intercept_)
output = lr_model.predict(X_test)
print("Mean squared error: %.3f" % mean_squared_error(y_test, output))
print("Coefficient of determination: %.3f" % r2_score(y_test, output))