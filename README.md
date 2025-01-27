# Heart Disease Prediction Programs

## Heart Disease Prediction Model  

**Description**:  
This Python program implements a Random Forest Classifier to predict heart disease based on the "heart.csv" dataset. The program performs data analysis, preprocessing, model training, and evaluation, including visualizations for insights.

### Dataset Information  
**Attribute Information**:  
- `age`: Age of the patient  
- `sex`: Gender of the patient (1 = male, 0 = female)  
- `chest pain type`: 4 types (values 1-4)  
- `resting blood pressure`: Blood pressure in mm Hg  
- `serum cholestoral in mg/dl`: Cholesterol levels  
- `fasting blood sugar > 120 mg/dl`: 1 if true, 0 if false  
- `resting electrocardiographic results`: Values 0, 1, or 2  
- `maximum heart rate achieved`: Maximum heart rate during exercise  
- `exercise induced angina`: 1 if yes, 0 if no  
- `oldpeak`: ST depression induced by exercise relative to rest  
- `slope of the peak exercise ST segment`: 3 types  
- `number of major vessels`: 0-3, colored by fluoroscopy  
- `thal`: 0 = normal, 1 = fixed defect, 2 = reversible defect  

**Note**: The names and social security numbers of patients were removed and replaced with dummy values.

---

### Key Steps in the Program:  
1. Load the dataset and perform basic analysis.  
2. Visualize correlations and the distribution of the target variable (heart disease).  
3. Prepare data, including splitting it into training and testing sets, and standardizing the features.  
4. Build and train a Random Forest Classifier.  
5. Evaluate the model performance with accuracy, classification report, confusion matrix, and feature importance.

### Output:  
- Accuracy score and evaluation metrics of the model.  
- **Visualizations**:  
  - Feature Correlation Heatmap  
  - Target Distribution  
  - Confusion Matrix  
  - Feature Importance Bar Chart  

---

## Program 1: Command-Line Heart Disease Predictor  

**Description**:  
This program analyzes a heart disease dataset (`heart.csv`) to train and evaluate a Random Forest Classifier. It includes functionality for exploratory data analysis (EDA), such as plotting feature correlations and visualizing the target variable distribution.  

### Key Features:  
- Reads and processes the dataset using Pandas.  
- Explores data statistics, missing values, and feature correlations with Seaborn and Matplotlib visualizations.  
- Trains a Random Forest Classifier using scikit-learn.  
- Displays the model's accuracy, classification report, and confusion matrix.  
- Shows feature importance in bar chart format.

### Usage:  
Run the script in a Python environment with `heart.csv` located in the same directory. The results are displayed in the console and through plots.

---

## Program 2: GUI-Based Heart Disease Predictor  

**Description**:  
This program provides a user-friendly graphical interface using `tkinter`. Users can load a dataset, train a Random Forest Classifier, and input patient attributes to predict whether the patient has heart disease.

### Key Features:  
- Interactive GUI for easy usage.  
- Allows users to load a heart disease dataset and train the model.  
- Input fields for patient attributes (e.g., age, sex, cholesterol level) with validation.  
- Displays prediction results interactively (Heart Disease: Yes/No).  
- Supports feature scaling and model preparation internally.

### Usage:  
Run the script to open the GUI application. Load the dataset through the "Load Dataset" button, input patient attributes, and click "Predict" to see the results.

---

## Differences Between the Two Programs  

| **Aspect**                 | **Command-Line Program**                      | **GUI Program**                          |
|----------------------------|-----------------------------------------------|------------------------------------------|
| **Interface**              | Command-line and plots                       | Graphical User Interface (GUI)          |
| **Interaction**            | Non-interactive, requires modifying the code | Interactive, user inputs via GUI        |
| **EDA and Visualization**  | Includes data exploration and visualizations | No EDA or visualizations                |
| **Prediction Workflow**    | Focused on dataset analysis and reporting    | Focused on real-time prediction         |
| **Ease of Use**            | Requires programming knowledge               | Accessible to non-technical users       |

**Conclusion**:  
- The **command-line program** is ideal for data scientists or developers needing detailed analysis and exploration of the dataset.  
- The **GUI program** is suited for end-users or clinicians seeking straightforward heart disease predictions without delving into code.  
