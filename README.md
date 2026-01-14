#  House Price AI: Machine Learning Predictor

**House Price AI** is an interactive web application that estimates residential property prices based on key features like square footage, build quality, and garage capacity.

Unlike static Jupyter Notebooks, this project deploys a **Linear Regression model** into a live, user-friendly interface, allowing users to tweak input parameters and see instant price predictions along with the model's accuracy metrics.

 **Live Demo:** [https://house-price-predictor-d235puue4xxatph2r25hdj.streamlit.app/]

---

###  Key Features

* **Real-Time Inference:** Instantly calculates the estimated price as you adjust sliders and inputs.
* **Transparent Metrics:** Displays the model's **R² Score (Accuracy)** dynamically to show how well the model fits the data.
* **Interactive Inputs:** Users can experiment with:
    * **Overall Material Quality** (1-10 Scale)
    * **Living Area** (sq ft)
    * **Garage Capacity** (Car count)
    * **Basement Size** & **Year Built**
* **Automated Training:** The app retrains the model on the fly using the latest dataset (`train.csv`) whenever the app loads, ensuring transparency in the modeling process.

###  Tech Stack

* **Frontend:** Streamlit (Python Web Framework)
* **Machine Learning:** Scikit-Learn (Linear Regression, Train-Test Splitting)
* **Data Processing:** Pandas (Data Cleaning & Feature Engineering)
* **Dataset:** Kaggle House Prices - Advanced Regression Techniques

###  How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/House-Price-Predictor.git](https://github.com/yourusername/House-Price-Predictor.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

---
*Note: This model is trained on a subset of the Ames Housing Dataset and is intended for educational and demonstration purposes.*
