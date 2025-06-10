# Wine Quality Prediction Project

![Screenshot of the application](path/to/your/screenshot.png)
*(Replace `path/to/your/screenshot.png` with the actual path to a screenshot of your working application. A good screenshot makes a huge difference!)*

## Project Overview

This project provides an interactive web application to predict the quality of red wines based on their physicochemical properties. Users can input various chemical attributes of a wine, and the system will predict whether its quality is considered "Good" or "Bad," along with an explanation of the contributing factors.

The backend is built with Flask and utilizes a pre-trained Machine Learning model. The frontend is a simple HTML/CSS/JavaScript interface for user interaction.

## Features

* **Wine Quality Prediction:** Predicts if a red wine is "Good" or "Bad" based on 11 physicochemical inputs.
* **Predefined Wine Examples:** Select from a list of popular red wines to see immediate predictions and reasoning.
* **Custom Wine Input:** Manually enter specific chemical properties to get a personalized prediction.
* **Detailed Reasoning:** Provides insights into which chemical properties positively or negatively influence the predicted quality.
* **Input Validation:** Ensures that entered values are within sensible ranges.

## Technologies Used

### Backend (Python)
* **Flask:** Web framework for building the API.
* **Scikit-learn:** For machine learning model (Decision Tree Classifier) and preprocessing (StandardScaler).
* **Pandas:** Data manipulation and handling.
* **NumPy:** Numerical operations.
* **Joblib:** For saving and loading the trained model and scaler.
* **Flask-CORS:** To handle Cross-Origin Resource Sharing.

### Frontend (Web)
* **HTML5:** Structure of the web page.
* **CSS3 (Tailwind CSS):** Styling and responsive design.
* **JavaScript:** Dynamic interactions, fetching predictions from the backend API.

## Project Structure