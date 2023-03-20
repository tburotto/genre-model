from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd
import pathlib

app = FastAPI(title= "Movie Genre Prediction")

dataframe = pd.read_csv("data/final.csv")
genre_list = ['Action', 'Adventure', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'History', 'Horror', 'Music', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'Western']
model = joblib.load(pathlib.Path('model/films-v1.joblib'))

class Movie(BaseModel):
    name: str

@app.post("/predict")
def predict(movie: Movie):
    aux = dataframe.loc[dataframe.Title.str.startswith(movie.name), (dataframe.columns != "Title") & (dataframe.columns != "genre")]
    if aux.shape[0] > 0:
        result = model.predict(aux)
        return {"genre": genre_list[result[0]]}
    else:
        return {"genre": "Movie not found!"}
