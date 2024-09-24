import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import re
pd.set_option("display.max_columns",None)
warnings.filterwarnings("ignore")

class House_Price_Class:
    
    def __init__(self, data):
        self.data2 = data
        
    def data_preprocessing(self):
        data = self.data2[["price", "price_per_sqft", "area", "areaWithType", "bedRoom", "bathroom", "additionalRoom", "facing","rating"]]
        data['facing'].fillna(data['facing'].mode()[0])
        data['plot_area_sqft'] = data['areaWithType'].apply(lambda x: re.findall(r'\d+', x)[0])
        data['plot_area_sqft'] = pd.to_numeric(data['plot_area_sqft'])
        data['average_rating'] = data['rating'].apply(self.extract_avg_rating)
        data["additionalRoom"] = data["additionalRoom"].apply(self.extract_additional_rooms)
        data.drop(["areaWithType", "facing", "rating"], axis=1, inplace=True)
        
        for col in data.columns:
            data[col].fillna(data[col].median(), inplace=True)  
        
        def remove_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            
        cleaned_data = remove_outliers(data, "price") 
        return cleaned_data
                   
        
    def extract_avg_rating(self, rating):
        if pd.isna(rating) or rating == "NaN":
            return None
        ratings = re.findall(r'(\d) out of 5', rating)
        if ratings:
            return sum(int(r) for r in ratings) / len(ratings)
            
    def extract_additional_rooms(self, rooms):
        room_types = ['servant room', 'store room', 'pooja room', 'study room', 'others']
        rooms_ = [1 if room in room_types else 0 for room in rooms.split(",")]
        return rooms_[0]
    
    def model_selection_and_prediction(self, X_train, X_test, y_train, y_test):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
        
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                           cv=5, n_jobs=-1, verbose=2, scoring='r2')
                
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        
        best_model = RandomForestRegressor(**best_params)
        best_model.fit(X_train, y_train)
        
        y_pred_best = best_model.predict(X_test)

        mse_best = mean_squared_error(y_test, y_pred_best)
        r2_best = r2_score(y_test, y_pred_best)
        
        print(f'Best Model Mean Squared Error: {mse_best}')
        print(f'Best Model R2 Score: {r2_best}')

        return r2_best
 
if __name__ == "__main__":
    data2 = pd.read_csv("data/house_cleaned.csv")
    House_Price_Class_obj = House_Price_Class(data2)
    cleaned_data = House_Price_Class_obj.data_preprocessing()
    y = cleaned_data.iloc[:,0]
    X = cleaned_data.iloc[:,1:] 
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    r2_best = House_Price_Class_obj.model_selection_and_prediction(X_train, X_test, y_train, y_test)
    
     