import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


class HeartDiseasePredictorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Heart Disease Predictor")
        self.master.geometry("600x600")

        self.model = None
        self.scaler = None
        self.dataset = None

        ttk.Label(master, text="Heart Disease Predictor", font=("Arial", 16)).pack(pady=10)

        self.load_button = ttk.Button(master, text="Load Dataset", command=self.load_dataset)
        self.load_button.pack(pady=5)

        ttk.Label(master, text="Input Features for Prediction:", font=("Arial", 12)).pack(pady=10)

        self.feature_inputs = {}
        feature_descriptions = {
            "age": "Age (Вік)",
            "sex": "Gender (Стать) [0=Female, 1=Male]",
            "cp": "Chest Pain Type (Тип болю в грудях) [1-4]",
            "trestbps": "Resting Blood Pressure (Тиск у стані спокою)",
            "chol": "Serum Cholesterol (Рівень холестерину в крові)",
            "fbs": "Fasting Blood Sugar (Рівень цукру натще) [0=No, 1=Yes]",
            "restecg": "Resting ECG Results (Результати ЕКГ) [0-2]",
            "thalach": "Max Heart Rate Achieved (Максимальна частота серцебиття)",
            "exang": "Exercise-Induced Angina (Стенокардія) [0=No, 1=Yes]",
            "oldpeak": "ST Depression (ST-депресія)",
            "slope": "Slope of Peak ST Segment (Нахил ST) [1-3]",
            "ca": "No. of Major Vessels (Кількість судин) [0-3]",
            "thal": "Thalassemia (Таласемія) [0-3]"
        }

        for feature, description in feature_descriptions.items():
            frame = ttk.Frame(master)
            frame.pack(fill="x", padx=20, pady=5)
            ttk.Label(frame, text=f"{description}:", width=40).pack(side="left")
            self.feature_inputs[feature] = ttk.Entry(frame)
            self.feature_inputs[feature].pack(side="right", fill="x", expand=True)

        self.predict_button = ttk.Button(master, text="Predict", command=self.predict)
        self.predict_button.pack(pady=10)

        self.result_label = ttk.Label(master, text="", font=("Arial", 12), foreground="blue")
        self.result_label.pack(pady=10)

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            self.dataset = pd.read_csv(file_path)
            self.prepare_model()
            messagebox.showinfo("Success", "Dataset loaded and model trained successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def prepare_model(self):
        if self.dataset is None:
            raise ValueError("Dataset not loaded!")

        X = self.dataset.drop(columns=['target'])
        y = self.dataset['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy Score:", acc)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

    def predict(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load the dataset and train the model first!")
            return

        try:
            input_data = {}
            for feature, entry in self.feature_inputs.items():
                value = float(entry.get())
                input_data[feature] = value

            input_df = pd.DataFrame([input_data])

            input_array = self.scaler.transform(input_df)

            prediction = self.model.predict(input_array)[0]

            if prediction == 1:
                self.result_label.config(text="Prediction: Heart Disease Detected (1)")
            else:
                self.result_label.config(text="Prediction: No Heart Disease (0)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict: {e}")


root = tk.Tk()
app = HeartDiseasePredictorApp(root)
root.mainloop()
