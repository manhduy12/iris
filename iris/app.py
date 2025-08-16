from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# 1. Load dữ liệu và huấn luyện model
df = pd.read_csv("Iris.csv")
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 2. Route chính
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        sl = float(request.form["sl"])
        sw = float(request.form["sw"])
        pl = float(request.form["pl"])
        pw = float(request.form["pw"])
        sample = [[sl, sw, pl, pw]]
        prediction = knn.predict(sample)[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
