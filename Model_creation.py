from sklearn.linear_model import LinearRegression
import joblib
from sklearn.model_selection import train_test_split
class model_creation:
    def __init__(self, data, y):
        self.data = data
        self.y = y

    def model_creation(self):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.y, test_size=0.3, random_state=55)
        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        joblib.dump(linreg, "model_save")
