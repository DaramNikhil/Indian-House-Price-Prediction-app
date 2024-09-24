# **Indian House Price Prediction App**

## **Introduction**

This is a machine learning web application built using **FastAPI** to predict house prices based on various features such as `price_per_sqft`, `area`, `bedroom`, `bathroom`, `additionalroom`, and `plot_sqft`. The application uses a pre-trained machine learning model to make predictions and provides an easy-to-use API for users to interact with.

## **Features**

1. **FastAPI** for a fast and interactive API.
2. **Swagger UI** for easy API testing.
3. **Scikit-learn** for machine learning model handling and feature scaling.
4. Pre-trained model for house price prediction.

## **Installation**

To install the Indian House Price Prediction project, follow these steps:

1. Clone the repository: **`git clone https://github.com/DaramNikhil/Indian-House-Price-Prediction-app.git`**
2. Navigate to the project directory: **`cd Indian-House-Price-Prediction-app`**
3. Install dependencies: **`pip install -r requirements.txt`**
4. Start the project: **`uvicorn app:app`**
5. Open your browser and navigate to: **`http://127.0.0.1:8000/docs`**

## **API Endpoints**

### POST `/predict`

**Request Body**:

```json
{
    "price_per_sqft": 20115.0,
    "area": 2610.0,
    "bedRoom": 3,
    "bathroom": 2,
    "additionalRoom": 1,
    "plot_area_sqft": 290,
    "average_rating": 4.25
}
```

**Response**

```
{
  "predicted_price": 4.811
}

```

## **License**

The Indian House Price Prediction project is released under the MIT License. See the **[LICENSE](https://github.com/DaramNikhil/Indian-House-Price-Prediction-app/blob/main/LICENSE)** file for details.

## **Contact**

-   **Author**: Daram Nikhil
-   **Email**: [nikhildaram51@gmail.com](mailto:nikhildaram51@gmail.com)
-   **GitHub**: [DaramNikhil](https://github.com/DaramNikhil)
-   **LinkedIn**: [daramnikhil](https://www.linkedin.com/in/daramnikhil)
