import streamlit as st; from PIL import Image
import pandas as pd
import numpy as np
from sklearn.ensemble         import RandomForestRegressor
from sklearn.linear_model     import LinearRegression
from sklearn.tree             import DecisionTreeRegressor
from sklearn.svm              import SVR
from sklearn.ensemble         import GradientBoostingRegressor
from sklearn.neural_network   import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import metrics
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
import joblib

st.write("Note: The predicted values are based on machine learning models deveoped by the author. "
         "The research in preliminary stage and the values are just for giving a rough idea of the properties of Self Compacting Recycled Aggregate Concretes"
         "The results shall not be considered as final and experimental assessment of properties shall be done in practice.")
image1=Image.open('developedby.png')
st.image(image1)
