import pandas as pd
import numpy as np
def create_dummies( df, colname ):
    col_dummies = pd.get_dummies(df[colname], prefix=colname)
    col_dummies.drop(col_dummies.columns[0], axis=1, inplace=True)
    df = pd.concat([df, col_dummies], axis=1)
    df.drop( colname, axis = 1, inplace = True )
    return df
def outliercapping(x):
    x=x.clip(upper=x.quantile(0.99))
    x=x.clip(lower=x.quantile(0.01))
    return x
class Preprocessing:
    def __init__(self,data):
        self.data=data
    def preprocessing(self):
        self.data.drop("Item_Identifier",axis=1,inplace=True)
        dfnumerical=self.data[["Item_Weight","Item_Visibility","Item_MRP","Outlet_Establishment_Year","Item_Outlet_Sales"]]
        dfcat=self.data[["Item_Fat_Content","Item_Type","Outlet_Identifier","Outlet_Size","Outlet_Location_Type","Outlet_Type"]]
        #Outlier treatment
        dfnumerical=dfnumerical.apply(outliercapping)
        #Missing values Treatment
        dfnumerical.Item_Weight=dfnumerical.Item_Weight.fillna(value=dfnumerical.Item_Weight.mean())
        dfcat.Outlet_Size=dfcat.Outlet_Size.fillna(value=dfcat.Outlet_Size.mode()[0])
        y=np.log(dfnumerical["Item_Outlet_Sales"])
        dfnumerical.drop("Item_Outlet_Sales",axis=1,inplace=True)
        dfnumerical.drop("Outlet_Establishment_Year",axis=1,inplace=True)
        dfcat.drop("Outlet_Identifier",axis=1,inplace=True)
        dfcat["Outlet_Size"]=dfcat["Outlet_Size"].map({"Medium":2,"High":3,"Small":1})
        dfcat["Item_Fat_Content"]=dfcat["Item_Fat_Content"].replace({"LF":"Low Fat","low fat":"Low Fat"})
        dfcat["Item_Fat_Content"]=dfcat["Item_Fat_Content"].replace("reg","Regular")
        dfcat["Item_Fat_Content"]=dfcat["Item_Fat_Content"].map({"Low Fat":1,"Regular":0})
        dfcat["Outlet_Location_Type"]=dfcat["Outlet_Location_Type"].map({"Tier 1":1,"Tier 2":2,"Tier 3":3})
        dfcat= create_dummies( dfcat,"Outlet_Type")
        dfcat.drop("Item_Type",axis=1,inplace=True)
        df=pd.concat([dfnumerical,dfcat],axis=1)
        return(df,y)

