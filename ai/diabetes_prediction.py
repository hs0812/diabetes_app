import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from PIL import Image
from sklearn.metrics import confusion_matrix
import json

# Function to load translations based on language code
def load_translation(language_code):
    if language_code == 'zh':
        with open("C:\\Users\\User\\Desktop\\ai\\zh.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    elif language_code == 'ms':
        with open("C:\\Users\\User\\Desktop\\ai\\ms.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {
            "title": "Diabetes Prediction",
            "title_descriptions": "This app uses a machine learning model to predict whether you may have diabetes based on various health-related factors. Please provide your details, and our model will analyze the data and offer a prediction.",
            "personal_details": "Personal Details",
            "gender_label": "Gender",
            "age_label": "Age",
            "bmi_label": "Body Mass Index (BMI)",
            "health_indicators": "Health Indicators",
            "highchol_label": "High Cholesterol",
            "highbp_label": "High Blood Pressure",
            "physactivity_label": "Physical Activity",
            "general_health_label": "General Health",
            "mental_health_label": "Mental Health (Last 30 days)",
            "physical_health_label": "Physical Health (Last 30 days)",
            "submit_button": "Submit",
            "summary_label": "Show summary of your inputs",
            "diabetic_prediction": "The model predicts: Diabetic",
            "non_diabetic_prediction": "The model predicts: Non-Diabetic",
            "confidence_level": "Confidence Level",
            "about_model_title": "About the Model",
            "about_model_description": "This machine learning model was trained on a large dataset of individuals' health data to predict the likelihood of having diabetes. It uses Logistic Regression to analyze the following features:",
            "about_model_features": "- Age\n- BMI (Body Mass Index)\n- High Cholesterol\n- High Blood Pressure\n- Physical Activity\n- General Health Rating\n- Days of poor mental and physical health in the last 30 days.",
            "cdc_info": "For more information on diabetes, visit the [CDC website](https://www.cdc.gov/diabetes).",
            # Add missing summary translations
            "gender_summary": "Gender",
            "age_summary": "Age",
            "bmi_summary": "Body Mass Index",
            "highchol_summary": "High Cholesterol",
            "highbp_summary": "High Blood Pressure",
            "physactivity_summary": "Physical Activity",
            "general_health_summary": "General Health",
            "mental_health_summary": "Mental Health (Last 30 days)",
            "physical_health_summary": "Physical Health (Last 30 days)"
        }

# Load the trained models, imputer, and scaler from the pickle file
with open('diabetes_models.pkl', 'rb') as file:
    models_data = pickle.load(file)

# Extract models, imputer, and scaler from the loaded dictionary
models = models_data["models"]
imputer = models_data["imputer"]
scaler = models_data["scaler"]

# Sidebar for language selection
language = st.sidebar.selectbox("Pilih Bahasa / 选择语言 / Select Language", 
                                ("English", "Chinese", "Malay"), 
                                key="language_selection")


# Load the appropriate translations based on the language selection
translations = load_translation('zh' if language == "Chinese" else 'ms' if language == "Malay" else 'en')

# Set up the Streamlit interface
st.title(translations['title'])
st.markdown(translations['title_descriptions'])

# Model selection sidebar
st.sidebar.title("Model Selection")
model_names = {
    "Logistic Regression": "logreg",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gbc_model",
    "K-Nearest Neighbors": "knn",
    "Support Vector Machine": "svm_model",
    "Naive Bayes": "nb_model"
}
selected_model_name = st.sidebar.selectbox("Choose a model:", list(model_names.keys()))

# Retrieve the selected model based on user input
selected_model = models[model_names[selected_model_name]]

# Create expandable sections for more interactive input experience
# Real-Time Input Section
with st.expander(translations['personal_details']):
    st.write("Provide your personal information for a more accurate prediction:")
    gender = st.radio(translations['gender_label'], options=["Male", "Female"], index=0)
    age = st.number_input(translations['age_label'], min_value=0, max_value=120, value=25)
    if age < 0 or age > 120:
        st.error("Age must be between 0 and 120.")

    bmi = st.number_input(translations['bmi_label'], min_value=0.0, max_value=50.0, value=25.0, step=1.0)
    if bmi < 10 or bmi > 50:
        st.error("BMI must be between 10 and 50.")

with st.expander(translations['health_indicators']):
    st.write("Provide details about your health:")
    highchol = st.radio(translations['highchol_label'], options=[0, 1], index=0)
    highbp = st.radio(translations['highbp_label'], options=[0, 1], index=0)
    physactivity = st.radio(translations['physactivity_label'], options=[0, 1], index=0)

with st.expander(translations['general_health_label']):
    genhlth = st.slider(translations['general_health_label'], min_value=1, max_value=5, value=3)
    menthlth = st.slider(translations['mental_health_label'], min_value=0, max_value=30, value=5)
    physhlth = st.slider(translations['physical_health_label'], min_value=0, max_value=30, value=5)

# Add the checkbox to show the summary of user inputs
if st.checkbox(translations['summary_label']):
    summary_data = {
        translations['gender_summary']: gender,
        translations['age_summary']: age,
        translations['bmi_summary']: bmi,
        translations['highchol_summary']: highchol,
        translations['highbp_summary']: highbp,
        translations['physactivity_summary']: physactivity,
        translations['general_health_summary']: genhlth,
        translations['mental_health_summary']: menthlth,
        translations['physical_health_summary']: physhlth
    }
    st.write(pd.DataFrame([summary_data]))

# Tabs for Model Comparison, Feature Importance, and ROC Curve
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Model Comparison", "Feature Importance", "ROC Curve", "Diabetes and Non-diabetes Comparison", "Confusion Matrix"])

# Tab 1: Model Comparison
with tab1:
    st.subheader("Model Performance Comparison")
    
    # Example performance metrics
    models_list = ['Gradient Boosting', 'Naive Bayes', 'SVM', 'Logistic Regression', 'KNN', 'Random Forest']
    accuracy = [0.75, 0.72, 0.74, 0.74, 0.71, 0.71]
    precision = [0.75, 0.72, 0.74, 0.74, 0.71, 0.71]
    recall = [0.75, 0.72, 0.74, 0.74, 0.71, 0.71]
    f1_score = [0.75, 0.72, 0.74, 0.74, 0.71, 0.71]

    # Create a figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    df_metrics = pd.DataFrame({
        'Model': models_list,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    })
    df_metrics.set_index('Model', inplace=True)

    # Plot the data
    df_metrics.plot(kind='bar', ax=ax)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

# Tab 2: Feature Importance (for tree-based models or logistic regression)
with tab2:
    st.subheader("Feature Importance")

    try:
        if hasattr(selected_model, 'coef_'):
            # Add 'Gender' to the list of feature names
            feature_names = ['Age', 'BMI', 'High Cholesterol', 'Physical Activity', 'General Health', 
                             'Mental Health (30 days)', 'Physical Health (30 days)', 'High Blood Pressure', 'Gender']

            # Coefficients from Logistic Regression (selected_model)
            coefficients = selected_model.coef_[0]  # Coefficients for the features

            # Create a DataFrame for visualization
            feature_importances = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
            feature_importances = feature_importances.sort_values(by='Coefficient', ascending=False)

            # Plot the coefficients using Seaborn
            plt.figure(figsize=(10, 6))
            sns.barplot(x="Coefficient", y="Feature", data=feature_importances)
            plt.title(f"Feature Importance (Logistic Regression Coefficients) for {selected_model_name}")
            st.pyplot(plt)

        elif hasattr(selected_model, 'feature_importances_'):
            # Add 'Gender' to the list of feature names for tree-based models
            feature_names = ['Age', 'BMI', 'High Cholesterol', 'Physical Activity', 'General Health', 
                             'Mental Health (30 days)', 'Physical Health (30 days)', 'High Blood Pressure', 'Gender']

            importances = selected_model.feature_importances_
            feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

            # Plot feature importance for tree-based models
            plt.figure(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=feature_importances)
            plt.title(f"Feature Importance for {selected_model_name}")
            st.pyplot(plt)

        else:
            st.write("Feature importance is not available for this model.")

    except AttributeError:
        st.write("Feature importance is not available for this model.")


# Tab 3: ROC Curve for multiple models
with tab3:
    st.subheader("ROC Curve for Multiple Models")

    # Load the image file
    image = Image.open('roc_curve.png')

    # Display the image in Streamlit
    st.image(image, caption='ROC Curve', use_column_width=True)

# Tab 4:
with tab4:
    st.subheader("Diabetes and Non-diabetes Metrics Comparison")
    
    # Define the methods and performance metrics (example data, replace with your actual values)
    methods = ["LR",  "RF", "SVM", "KNN", "GB", "NB"]

    # Example data for Accuracy, Precision, Recall, and F1-Score
    accuracy = [0.74, 0.71, 0.74, 0.71, 0.75, 0.72]
    precision_normal = [0.76, 0.72, 0.77, 0.73, 0.78, 0.71]
    precision_diabetes = [0.73, 0.70, 0.72, 0.70, 0.73, 0.73]
    recall_normal = [0.72, 0.68, 0.69, 0.68, 0.70, 0.74]
    recall_diabetes = [0.77, 0.74, 0.79, 0.74, 0.80, 0.70]
    f1_normal = [0.74, 0.70, 0.73, 0.81, 0.74, 0.73 ]
    f1_diabetes = [0.75, 0.72, 0.75, 0.72, 0.76, 0.72]

    # Create subplots for accuracy, precision, recall, and F1-score
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Define width for side-by-side bars
    bar_width = 0.35
    index = np.arange(len(methods))

    # Accuracy plot (single plot, no comparison)
    axs[0, 0].bar(index, accuracy, bar_width, color='lightgreen')
    axs[0, 0].set_title('Accuracy')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_xticks(index)
    axs[0, 0].set_xticklabels(methods)
    axs[0, 0].set_ylim([0, 1])

    # Precision plot (side-by-side)
    axs[0, 1].bar(index, precision_normal, bar_width, color='orange', label='Normal')
    axs[0, 1].bar(index + bar_width, precision_diabetes, bar_width, color='lightgreen', label='Diabetes')
    axs[0, 1].set_title('Precision')
    axs[0, 1].set_ylabel('Precision')
    axs[0, 1].set_xticks(index + bar_width / 2)
    axs[0, 1].set_xticklabels(methods)
    axs[0, 1].set_ylim([0, 1])
    axs[0, 1].legend()

    # Recall plot (side-by-side)
    axs[1, 0].bar(index, recall_normal, bar_width, color='orange', label='Normal')
    axs[1, 0].bar(index + bar_width, recall_diabetes, bar_width, color='lightgreen', label='Diabetes')
    axs[1, 0].set_title('Recall')
    axs[1, 0].set_ylabel('Recall')
    axs[1, 0].set_xticks(index + bar_width / 2)
    axs[1, 0].set_xticklabels(methods)
    axs[1, 0].set_ylim([0, 1])
    axs[1, 0].legend()

    # F1-Score plot (side-by-side)
    axs[1, 1].bar(index, f1_normal, bar_width, color='orange', label='Normal')
    axs[1, 1].bar(index + bar_width, f1_diabetes, bar_width, color='lightgreen', label='Diabetes')
    axs[1, 1].set_title('F1-Score')
    axs[1, 1].set_ylabel('F1-Score')
    axs[1, 1].set_xticks(index + bar_width / 2)
    axs[1, 1].set_xticklabels(methods)
    axs[1, 1].set_ylim([0, 1])
    axs[1, 1].legend()

    # Adjust layout and display the plot
    plt.tight_layout()
    st.pyplot(fig)

# Tab 4: Confusion Matrix Comparison
with tab5:
    st.subheader("Confusion Matrix Comparison")

    # Load each image
    images = {
        "Logistic Regression": Image.open('C:\\Users\\User\\Desktop\\ai\\lr.png'),
        "Random Forest": Image.open('C:\\Users\\User\\Desktop\\ai\\rf.png'),
        "Gradient Boosting": Image.open('C:\\Users\\User\\Desktop\\ai\\gb.png'),
        "K-Nearest Neighbors": Image.open('C:\\Users\\User\\Desktop\\ai\\knn.png'),
        "Support Vector Machine": Image.open('C:\\Users\\User\\Desktop\\ai\\svm.png'),
        "Naive Bayes": Image.open('C:\\Users\\User\\Desktop\\ai\\nb.png'),
    }

    # Create a 2x3 layout for displaying the confusion matrix images
    cols = st.columns(2)  # Two columns

    # Iterate over images and display them
    for i, (model_name, img) in enumerate(images.items()):
        col = cols[i % 2]  # Alternate between the two columns
        with col:
            st.image(img, caption=f'{model_name} Confusion Matrix', use_column_width=True)

    # Adjust for extra spacing
    st.markdown("<br><br>", unsafe_allow_html=True)

# Prediction section (predict button)
if st.button(translations['submit_button']):
    # Encode gender as 0 for Female and 1 for Male
    gender_encoded = 1 if gender == "Male" else 0

    # Prepare the input data as a numpy array (including gender_encoded as the missing feature)
    input_data = np.array([[age, bmi, highchol, physactivity, genhlth, menthlth, physhlth, highbp, gender_encoded]])

    # Impute and scale the input data
    input_data_imputed = imputer.transform(input_data)
    input_data_scaled = scaler.transform(input_data_imputed)

    # Make a prediction using the selected model
    prediction = selected_model.predict(input_data_scaled)

    # Check if the selected model has predict_proba for probability estimation
    if hasattr(selected_model, 'predict_proba'):
        prediction_proba = selected_model.predict_proba(input_data_scaled)

        # Output the prediction with more details
        if prediction[0] == 1:
            st.write(f"### {translations['diabetic_prediction']}")
            confidence = prediction_proba[0][1]  # Probability of class 1 (diabetic)
        else:
            st.write(f"### {translations['non_diabetic_prediction']}")
            confidence = prediction_proba[0][0]  # Probability of class 0 (non-diabetic)

        # Display prediction probability
        st.write(f"#### {translations['confidence_level']}: {confidence * 100:.2f}%")
    else:
        # Output prediction without confidence level if predict_proba is unavailable
        if prediction[0] == 1:
            st.write(f"### {translations['diabetic_prediction']}")
        else:
            st.write(f"### {translations['non_diabetic_prediction']}")

# Additional information section in the sidebar
st.sidebar.title(translations['about_model_title'])
st.sidebar.markdown(translations['about_model_description'])
st.sidebar.markdown(translations['about_model_features'])
st.sidebar.markdown(translations['cdc_info'])
