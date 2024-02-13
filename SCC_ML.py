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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import metrics
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
import joblib
import pickle
'''
CS = joblib.load('Stack3_CS.joblib')
SF =joblib.load('BG_SF.joblib')
T500= joblib.load('BG_T500.joblib')
VF =joblib.load('GBR_VF.joblib')
st.write('Self compacting recycled aggregate concrete Compressive Strength and fresh properties predictor:')
Cem=st.number_input('Cement and pozzolana content kg/cum')
FAsh=st.number_input('Fly Ash content in kg/cum')
FAgg=st.number_input('Fine Aggregates content in kg/cum')
NAgg=st.number_input('Normal Aggregates content in kg/cum')
RCA=st.number_input('Recycled coarse aggregates content in kg/cum')
Water=st.number_input('Water contnet in kg/cum')
Spz=st.number_input('Superplasticizer content in kg/cum')

if st.button('Predict properties'):
  BTa=((Cem + FAsh)/(FAgg + NAgg+RCA))
  FAshB=(FAsh/(Cem+FAsh))
  BFAgg=((Cem+FAsh)/FAgg)
  FACA=(FAgg/(NAgg+RCA))
  RACA=(RCA/NAgg)
  WB=(Water/(Cem+FAsh))
  LS=((Water+Spz)/(Cem+FAsh+FAgg+NAgg+RCA))
  SpB=((Water)/(Cem+FAsh))
  input1 = [BTa,FAshB,BFAgg,FACA,RACA,WB,LS,SpB]
  input1 = np.array(input1).reshape(1, -1)
  CStr = CS.predict(input1)
  SFlow = SF.predict(input1)
  T500time= T500.predict(input1)
  VFun = VF.predict(input1)

  st.write("Predcited compressive strength is:"+ str(CStr) + "MPa")
  if 520<SFlow<670:
    st.write("The concrete is likely to fall in EFNARC-SF1 Slump flow class (Slump flow value between 520mm to 700mm.)")
  else:
      if 670<SFlow<870:
        st.write("The concrete is likely to fall in EFNARC-SF2 Slump flow class (Slump flow value between 640mm to 800mm.)")
      else:
        if 770<SFlow<900:
          st.write("The concrete is likely to fall in EFNARC-SF3 Slump flow class (Slump flow value between 740mm to 900mm.)")
        else:
          if SFlow>900:
             st.write("The slump flow of concrete is expected to be higher than EFNARC-SF3 class, no class defined.")
          else:
             st.write("The slump flow of concrete is expected lesser than EFNARC-SF1 class. The concrete might not be self compacting.")

  if 0<VFun<8.5:
    st.write("The concrete is likely to fall in EFNARC-VF1 V-funnel class (V-funnel time <10s.)")
  else:
      if 8.5<VFun<27:
        st.write("The concrete is likely to fall in EFNARC-VF2 V-funnel class (V-funnel time between 7s to 27s.)")
      else:
        if VFun>27:
          st.write("The concrete is likely to take V-Funnel time more than 27s (beyond EFNARC-VF2 class).")
        else:
          st.write("Error: V-Funnel time could not be predicted.")
  st.write("Rough prediction of T500 time:"+str(T500time)+"s")
'''
st.write("Note: The predicted values are based on machine learning models deveoped by the author. "
         "The research in preliminary stage and the values are just for giving a rough idea of the properties of Self Compacting Recycled Aggregate Concretes"
         "The results shall not be considered as final and experimental assessment of properties shall be done in practice.")
image1=Image.open('developedby.png')
st.image(image1)
