from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score

X, y = load_iris(return_X_y=True)
# print("X is",X)
# print("y is",y)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=91)


lg_model = LogisticRegression(solver = 'liblinear')

lg_model.fit(X_train, y_train)

# model has learned to classify the irises?

# test the model
output = lg_model.predict(X_test)



a_score = accuracy_score(y_test, output)
r_score = recall_score(y_test, output, average = 'weighted')
p_score = precision_score(y_test, output,average = 'weighted')

print(f"accuracy: {a_score}")
print(f"recall: {r_score}")
print(f"precision: {p_score}")