# Retail-Analytics-And-Sales-Forecasting-2B765
 Retail Analytics & Sales Forecasting (ID: 2B765)

# 🛍️ Retail Analytics & Sales Forecasting

**Project ID:** 2B765  
**Team:** Strawberry Lab  
**Repository:** [Retail-Analytics-And-Sales-Forecasting-2B765](https://github.com/ShamScripts/Retail-Analytics-And-Sales-Forecasting-2B765)

---

## 📌 Overview

This project provides an end-to-end solution for analyzing retail sales data and forecasting future sales using advanced analytics and machine learning. It includes interactive dashboards, exploratory data visualizations, predictive modeling, and exportable reports to support business decision-making.

---

## 🔧 Features

- 📊 **Interactive Dashboards** for sales insights
- 🔍 **Advanced Sales Analysis** with feature importance and product/category exploration
- ⏱️ **Time Series Forecasting** with custom date range and forecast horizons
- 🔮 **Sales Prediction Tool** for both single and batch predictions
- 📄 **PDF & Excel Report Generation**
- 📈 Visuals powered by Plotly and Streamlit
- 📁 Modular structure for maintainability and expansion


---

## 📊 Dashboard Modules

| Module              | Description |
|---------------------|-------------|
| **Overview**        | Summary metrics and dataset preview |
| **Advanced Insights** | Feature importance, product analysis, outlet comparison |
| **Time Series**     | Sales trend visualization and forecasting |
| **Reports**         | Export customizable PDF/Excel reports |
| **Predict Sales**   | Predict sales for input attributes (form or batch upload) |

---

## 🧠 Machine Learning Model

- **Model:** XGBoost Regressor
- **Features Used:**  
  `Item_MRP`, `Outlet_Identifier`, `Outlet_Size`, `Outlet_Type`, `Outlet_Age`
- **Evaluation:**  
  Custom feature importance scoring and sales classification thresholds

---

## 📦 Installation

```bash
git clone https://github.com/ShamScripts/Retail-Analytics-And-Sales-Forecasting-2B765.git
cd Retail-Analytics-And-Sales-Forecasting-2B765
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧪 Sample Prediction

You can test predictions by entering:
- `Item MRP`: 150
- `Outlet Type`: Supermarket Type2
- `Outlet Size`: Medium
- `Established`: 2005

Or upload a CSV for batch predictions with:
```
Item_MRP,Outlet_Identifier,Outlet_Size,Outlet_Type,Outlet_Establishment_Year
199.0,OUT017,Medium,Supermarket Type1,2004
```

---

## 📤 Deployment

Easily deployable on:
- **Render**
- **Streamlit Community Cloud**
- **Heroku / Docker (optional)**

---

## 📃 License

MIT License © SHAMBHAVI JHA
See `LICENSE` for more details.

---

## 🤝 Contributions

Pull requests and improvements are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📬 Contact

For inquiries, feel free to reach out via:

- [LinkedIn](https://www.linkedin.com/in/shamscript009)  
- [Gmail](mailto:f20230009@dubai.bits-pilani.ac.in)

