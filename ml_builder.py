import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MLModelBuilder(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.title("Machine Learning Model Builder")
        self.geometry("900x700")
        
        # Initialize a theme variable
        self.is_dark_mode = True

        # Variables to store dataset and model
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

        # Apply a custom style for ttk widgets
        self.style = ttk.Style()
        self.apply_dark_mode()

        # Create and layout the UI components
        self.create_widgets()

    def create_widgets(self):
        """Create all the UI components with better layout and styling."""

        # Theme switcher button
        self.theme_button = ttk.Button(self, text="Switch to Light Mode", command=self.toggle_theme)
        self.theme_button.pack(pady=10)

        # Model Selection
        self.model_frame = ttk.Frame(self)
        self.model_frame.pack(pady=10)

        self.model_label = ttk.Label(self.model_frame, text="Select Model:")
        self.model_label.grid(row=0, column=0, padx=10)

        self.model_options = [
            "Linear Regression", "Decision Tree", "Random Forest", 
            "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)", "Neural Network (MLP)"
        ]
        self.selected_model = tk.StringVar()
        self.model_menu = ttk.Combobox(self.model_frame, textvariable=self.selected_model, values=self.model_options, state="readonly")
        self.model_menu.grid(row=0, column=1, padx=10)
        self.model_menu.set("Select a model")  # Default text

        # Hyperparameters Input
        self.param_frame = ttk.Frame(self)
        self.param_frame.pack(pady=10)

        self.param_label = ttk.Label(self.param_frame, text="Hyperparameters:")
        self.param_label.grid(row=0, column=0, columnspan=2)

        self.max_depth_label = ttk.Label(self.param_frame, text="Max Depth:")
        self.max_depth_label.grid(row=1, column=0, sticky="e", padx=10, pady=5)
        self.max_depth_entry = ttk.Entry(self.param_frame)
        self.max_depth_entry.grid(row=1, column=1, padx=10, pady=5)

        self.n_estimators_label = ttk.Label(self.param_frame, text="Number of Estimators:")
        self.n_estimators_label.grid(row=2, column=0, sticky="e", padx=10, pady=5)
        self.n_estimators_entry = ttk.Entry(self.param_frame)
        self.n_estimators_entry.grid(row=2, column=1, padx=10, pady=5)

        # Load Dataset Button
        self.load_button = ttk.Button(self, text="Load Dataset (CSV)", command=self.load_dataset)
        self.load_button.pack(pady=10)

        # Train Model Button
        self.train_button = ttk.Button(self, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        # Output Text Area for displaying results
        self.output_area = tk.Text(self, height=10, width=100, bg="#34495E", fg="#ECF0F1", font=('Consolas', 12))
        self.output_area.pack(pady=20)

    def toggle_theme(self):
        """Toggle between dark mode and light mode."""
        if self.is_dark_mode:
            self.apply_light_mode()
        else:
            self.apply_dark_mode()
        self.is_dark_mode = not self.is_dark_mode
        self.theme_button.config(text="Switch to Light Mode" if not self.is_dark_mode else "Switch to Dark Mode")

    def apply_dark_mode(self):
        """Apply the dark mode theme."""
        self.configure(bg='#2C3E50')
        self.style.configure("TLabel", background='#2C3E50', foreground='#ECF0F1', font=('Helvetica', 12))
        self.style.configure("TButton", background='#34495E', foreground='#ECF0F1', font=('Helvetica', 10, 'bold'))
        self.style.configure("TCombobox", font=('Helvetica', 12))
        self.style.configure("TEntry", font=('Helvetica', 12))

    def apply_light_mode(self):
        """Apply the light mode theme."""
        self.configure(bg='#ECF0F1')
        self.style.configure("TLabel", background='#ECF0F1', foreground='#2C3E50', font=('Helvetica', 12))
        self.style.configure("TButton", background='#BDC3C7', foreground='#2C3E50', font=('Helvetica', 10, 'bold'))
        self.style.configure("TCombobox", font=('Helvetica', 12))
        self.style.configure("TEntry", font=('Helvetica', 12))

    def load_dataset(self):
        """Load a dataset from a CSV file."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.output_area.insert(tk.END, f"Loading dataset from {file_path}...\n")
            data = pd.read_csv(file_path)
            self.output_area.insert(tk.END, f"Dataset loaded. Shape: {data.shape}\n")

            # Assume the last column is the target variable
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
            self.output_area.insert(tk.END, "Dataset split into train and test sets.\n")
        else:
            self.output_area.insert(tk.END, "Dataset loading canceled.\n")

    def train_model(self):
        """Train the selected model based on user input."""
        if self.X_train is None or self.X_test is None:
            self.output_area.insert(tk.END, "Please load a dataset first.\n")
            return

        model_name = self.selected_model.get()
        self.output_area.insert(tk.END, f"Training {model_name}...\n")

        if model_name == "Linear Regression":
            self.model = LinearRegression()
        elif model_name == "Decision Tree":
            max_depth = self.get_param(self.max_depth_entry)
            self.model = DecisionTreeClassifier(max_depth=max_depth)
        elif model_name == "Random Forest":
            n_estimators = self.get_param(self.n_estimators_entry)
            max_depth = self.get_param(self.max_depth_entry)
            self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        elif model_name == "Support Vector Machine (SVM)":
            self.model = SVC()
        elif model_name == "K-Nearest Neighbors (KNN)":
            self.model = KNeighborsClassifier()
        elif model_name == "Neural Network (MLP)":
            self.model = MLPClassifier()
        else:
            self.output_area.insert(tk.END, "Unknown model selected.\n")
            return

        # Train the model
        self.model.fit(self.X_train, self.y_train)

        # Predictions and evaluation
        if model_name == "Linear Regression":
            y_pred = self.model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            self.output_area.insert(tk.END, f"Model trained. Mean Squared Error: {mse}\n")
        else:
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            self.output_area.insert(tk.END, f"Model trained. Accuracy: {accuracy * 100:.2f}%\n")
            self.plot_confusion_matrix(self.y_test, y_pred)

    def get_param(self, entry):
        """Helper method to get hyperparameter value from the entry field."""
        value = entry.get()
        if value.isdigit():
            return int(value)
        return None

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot a confusion matrix after training a classification model."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")
        plt.show()


if __name__ == "__main__":
    app = MLModelBuilder()
    app.mainloop()
