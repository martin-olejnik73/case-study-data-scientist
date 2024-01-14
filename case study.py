import pandas as pd 
import matplotlib.pyplot as plt

data1 = pd.read_csv("customer-spending-1.csv")
data2 = pd.read_csv("customer-spending-2.csv")
data3 = pd.read_csv("customer-spending-3.csv")

def general_analysis(data):
    column_info_data = {}
    columns_with_nulls_data = {}
    
    for column in data.columns:
        column_info_data[column] = {}    
        num_unique_entries = data[column].nunique()
        column_info_data[column]["num_unique_entries"]=num_unique_entries
    
        num_null_entries = data[column].isnull().sum()
        column_info_data[column]["num_null_entries"] = num_null_entries
    
        data_types = data[column].apply(type).unique()
        column_info_data[column]["data_types"] = data_types
    	
        if num_null_entries > 0:
            columns_with_nulls_data[column] =  num_null_entries
        
    return column_info_data,columns_with_nulls_data

def column_stats(data):
    result_dict = {}
    columns_to_ignore = ["user_id","prod_id","revenue_usd"]
    data = data.drop(columns=columns_to_ignore)
    
    for column in data.columns:
        entry_counts = data[column].value_counts().to_dict()    
        result_dict[column] = entry_counts    
        
    return result_dict
    
def plot_column_stats(data,columns_to_plot,name):
    for col in columns_to_plot:
        col_name = dict(sorted(data[col].items()))
        plt.bar(col_name.keys(),col_name.values())
        plt.ylabel("number of entries")
        plt.xlabel(col)
        plt.xticks(list(data[col].keys()))
        plt.title(f"distribution of {col} of {name}")
        plt.show()


def features(data):    
    data = data.sort_values(by="revenue_usd",ascending=True)
    data["previous_revenue"] = data["revenue_usd"].shift(1)
    data = data.reset_index()
    data["previous_revenue"][0] = data["revenue_usd"][0]
    data["price"] = data["revenue_usd"] - data["previous_revenue"]
    data["not_free"] = data["price"]>0	
    data = data.drop(columns="index")
	
    grouped_no_product = data.groupby(['sex', 'age_cat', "credit_status_cd", 'edcution_cat', 'years_in_residence', 'car_ownership']).agg({'price': ['sum', 'count'],'not_free': ['sum']}).reset_index()
    grouped_no_product.columns = ['_'.join(col).strip() for col in grouped_no_product.columns.values]   
    grouped_no_product["ratio"] = grouped_no_product["not_free_sum"]/grouped_no_product["price_count"]

    nonzero_no_product = grouped_no_product[grouped_no_product["ratio"]>0]
    nonzero_no_product["average_price"] = nonzero_no_product["ratio"]*nonzero_no_product["price_sum"]
    nonzero_no_product=nonzero_no_product.sort_values(by="average_price",ascending=False)
    nonzero_no_product=nonzero_no_product.reset_index()
    nonzero_no_product = nonzero_no_product.drop(columns="index")
	
    return data,grouped_no_product,nonzero_no_product

def products(data,topdata):    
    fin = {}
    prods = {}
    cols = ["sex","age_cat","credit_status_cd","edcution_cat","years_in_residence","car_ownership"]
    for j in range(0,len(topdata)):
        hel = data.copy()
        for i in range(0,len(cols)-1,2):
            c1 = hel[cols[i]] == topdata[cols[i]+"_"][j]
            c2 = hel[cols[i+1]] == topdata[cols[i+1]+"_"][j]
            hel = hel[c1 & c2]
        fin[j] = hel
        prods[j] = fin[j]["prod_id"].unique()
    return fin,prods

def incorrect(data):
    id_age = data[["user_id","age_cat"]]
    id_age["nunique"]=id_age.groupby('user_id')['age_cat'].transform('nunique')
    id_age = id_age.sort_values(by="nunique",ascending=False)
    nid_age_mismatch = len(set(id_age[id_age["nunique"]>1]["user_id"]))
    return nid_age_mismatch
	
data1_info, data1_columns_with_nulls = general_analysis(data1)
data2_info, data2_columns_with_nulls = general_analysis(data2)
data3_info, data3_columns_with_nulls = general_analysis(data3)

column_info_data1 = column_stats(data1)
column_info_data2 = column_stats(data2)
column_info_data3 = column_stats(data3)

data1_add,data1_grouped,data1_nonzero = features(data1)
data2_add,data2_grouped,data2_nonzero = features(data2)
data3_add,data3_grouped,data3_nonzero = features(data3)


top5_data1 = data1_nonzero.head()
not_free_prods1 = data1_add[data1_add["not_free"]==True]
top5_alldata1,prods_data1 = products(not_free_prods1,top5_data1)

top5_data2 = data2_nonzero.head()
not_free_prods2 = data2_add[data2_add["not_free"]==True]
top5_alldata2,prods_data2 = products(not_free_prods2,top5_data2)

top5_data3 = data3_nonzero.head()
not_free_prods3 = data3_add[data3_add["not_free"]==True]
top5_alldata3,prods_data3 = products(not_free_prods3,top5_data3)


columns_to_plot = ["sex","age_cat","credit_status_cd","edcution_cat","years_in_residence","car_ownership","prod_cat_1","prod_cat_2","prod_cat_3"]
# plot_column_stats(column_info_data1,columns_to_plot,"data1")
# plot_column_stats(column_info_data1,columns_to_plot,"data2")
# plot_column_stats(column_info_data1,columns_to_plot,"data3")

nid_age_data1 = incorrect(data1)
nid_age_data2 = incorrect(data2) 
nid_age_data3 = incorrect(data3) 


from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score


data = data1_add.copy()
data = data.sort_values(by="revenue_usd",ascending=True)
data["previous_revenue"] = data["revenue_usd"].shift(1)
data = data.reset_index()
data["previous_revenue"][0] = data["revenue_usd"][0]
data["price"] = data["revenue_usd"] - data["previous_revenue"]
data["not_free"] = data["price"]>0	
data = data.drop(columns="index")
data["not_free"]=data["not_free"].replace({True: 1, False: 0})
data = data.drop(columns=["prod_cat_1","prod_cat_2","prod_cat_3","revenue_usd","previous_revenue","price"])

grouped_data = data.groupby(['sex','age_cat', "credit_status_cd", 'edcution_cat', 'years_in_residence', 'car_ownership']).agg({'not_free': ['sum']}).reset_index()
grouped_data.columns = ['_'.join(col).strip() for col in grouped_data.columns.values]   
grouped_data["not_free_sum"] = grouped_data["not_free_sum"]>sum(grouped_data["not_free_sum"])/len(grouped_data["not_free_sum"])	
grouped_data["not_free_sum"]=grouped_data["not_free_sum"].replace({True: 1, False: 0})


data_encoded = pd.get_dummies(grouped_data, columns=["sex_", "age_cat_","edcution_cat_","years_in_residence_"])
data_encoded["not_free_sum"] = grouped_data["not_free_sum"]

X = data_encoded.drop("not_free_sum", axis=1)
y = data_encoded["not_free_sum"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=73)

xgb_model = XGBClassifier()

param_grid = {
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [10, 15, 20],
    'n_estimators': [150, 300 ]
}
scoring_metric = make_scorer(f1_score)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring=scoring_metric, cv=3)


grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy: {accuracy}")


