from sklearn import preprocessing

class CategoricalFeatures:
    """
    - # label encoding
    - one hot encoding
    - binarization
    """ 

    def __init__(self,df,categorical_features,encoding_type='label',handle_na=False):
        """
        df: pandas dataframe
        categorical_features: list of column names, eg ['bin_0', "ord_1", "nom_0"]
        encoding_type: label, binary, ohe
        handle_na: True / False
        """
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()


        
        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:,c] = self.df.loc[:,c].astype(str).fillna("-999999")

        self.output_df = self.df.copy(deep=True)            


    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()            
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df

    def _label_binarization(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()            
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values)
            self.output_df = self.output_df.drop(c,axis=1)
            for j in range(val.shape[1]):
                new_col_name = c+ f"__bin_{j}"
                self.output_df[new_col_name] = val[:, j]                        
            self.binary_encoders[c] = lbl          
        return self.output_df

    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:,c] = dataframe.loc[:,c].astype(str).fillna("-999999")
        if self.enc_type=='label':
            for c,lbl in self.label_encoders.items():                
                dataframe.loc[:,c] = lbl.transform(dataframe.loc[:,c])  
            return dataframe          
        elif self.enc_type=='binary':
            for c,lbl in self.binary_encoders.items():                                        
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c,axis=1)
                for j in range(val.shape[1]):
                    new_col_name = c+ f"__bin_{j}"
                    dataframe[new_col_name] = val[:, j]
            return dataframe                                                             
        else :
            raise Exception("Encoding type not understood")
        


    def fit_transform(self): 
        
        if self.enc_type=='label':
            return self._label_encoding()
        elif self.enc_type=='binary':
            return self._label_binarization()           
        else:
            raise Exception("Encoding type not understood")
        


if __name__=='__main__':
    import pandas as pd
    df = pd.read_csv('../input/train_cat.csv')#.head(500)
    df_test = pd.read_csv('../input/test_cat.csv')#.head(500)
    print(df.head())
    print(df_test.head())

    train_idx = df["id"].values

    df_test["target"] = -1
    test_idx  = df_test["id"].values
    

    full_data = pd.concat([df,df_test])
    cols = [c for c in full_data.columns if c not in ["id", "target"]]
    print(cols)
    cat_feats = CategoricalFeatures(full_data,
                                    categorical_features=cols,
                                    encoding_type='label',
                                    handle_na=True)
    
    full_data_transformed = cat_feats.fit_transform()

    train_transformed = full_data_transformed[full_data_transformed["id"].isin(train_idx)].reset_index(drop=True)
    test_transformed = full_data_transformed[full_data_transformed["id"].isin(test_idx)].reset_index(drop=True)
    # train_transfomred = cat_feats.fit_transform()
    # test_transformed = cat_feats.transform(df_test)
    print(train_transformed.head())
    print(test_transformed.head())