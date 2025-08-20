# Ads Analytics

**Author:** Mateus Pereira da Silva  
**License:** MIT

## Overview

**Ads Analytics** is a modular and scalable solution for predictive analytics on digital advertising data. Designed for reproducibility and flexibility, the project features robust data pipelines, supervised learning models, and interactive interfaces for rapid experimentation and insight generation.

---

## Architecture

### Data Pipeline

1. **Data Ingestion:** Loads raw or synthetic datasets.  
2. **Schema Validation:** Ensures the dataset conforms to the expected schema.  
3. **Data Cleaning:** Deduplicates and fills missing values.  
4. **Feature Engineering:** Creates new features and applies encoding.  
5. **Gold Dataset Validation:** Confirms the processed dataset meets quality standards.  
6. **Storage:** Saves the clean and validated dataset for modeling.

### Model Training & Forecast

1. **Sort by Date:** Orders data chronologically.  
2. **Temporal Split:** Divides data into training, testing, and forecasting sets.  
3. **Model Initialization:** Selects and configures the predictive model.  
4. **Training:** Fits the model on the training data.  
5. **Forecasting:** Generates predictions for testing and future datasets.  
6. **Artifact Storage:** Saves trained models and forecast outputs.

### Interactive Streamlit Application

- **Model Selection:** Users can choose between Decision Tree, XGBoost, Random Forest, LightGBM, or CatBoost.  
- **Training & Evaluation:** Trains the selected model and evaluates its performance.  
- **Visualizations:** Displays prediction plots, error metrics, and user segmentation.  
- **A/B Testing:** Supports controlled experiments for comparative analysis.

---

## Technologies

- **Languages:** Python  
- **Frameworks:** Polars, Streamlit  
- **Models:** XGBoost, CatBoost, LightGBM, Random Forest, Decision Tree  
- **Libraries:** scikit-learn, Plotly, Numpy
- **Storage:** Parquet  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/tzpereira/ads-analytics.git
cd ads-analytics
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the Data Pipeline

```bash
python pipeline.py
```

### Launch the Interactive App

```bash
streamlit run app/main.py
```

---

## Contributing

Contributions are welcome! Feel free to open issues or pull requests.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

