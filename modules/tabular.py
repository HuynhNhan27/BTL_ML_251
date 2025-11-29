# Tabular

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import r2_score, f1_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

########################## EDA #############################

def dataset_info(df: pd.DataFrame):
    '''
    Print Pandas.DataFrame info like: len, feature, shape, columns name, dtype
    '''
    print("\nDataset Info:")
    print(f"Total samples: {len(df):,}")
    print(f"Features: {df.shape[1]}")
    print(f"Data Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print("Dtypes:\n", df.dtypes)

def missing_report(df: pd.DataFrame):
    missing_df = df.isna().sum().sort_values(ascending=False)
    print("Missing Value:")
    print(pd.DataFrame({"Missing" : missing_df, "Percent" : (missing_df / len(df) * 100).round(2)}))

def column_report(df: pd.DataFrame):
    print("Column Report:")
    print(pd.DataFrame({"Dtypes" : df.dtypes, "Nunique" : df.nunique()}))

########################## End of EDA #############################


################################# Descriptive Statistics #######################################

################# Numeric ########################

def numeric_stat_report(df: pd.DataFrame):
    numeric_feature = list(df.select_dtypes(include=["number"]).columns)
    return numeric_feature, df.describe(include=["number"])

def numeric_plot(df: pd.DataFrame, config: dict, plot="histogram"):
    if config is None:
        config = {
            "normal": list(df.select_dtypes(include=["number"]).columns)
        }

    numeric_feature = [(transform, col) for transform, columns in config.items() for col in columns]
    print(numeric_feature)

    cols = 3
    rows = len(numeric_feature) // cols + 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.reshape(rows, cols)
    for i, (transform, col) in enumerate(numeric_feature):
        row_idx, col_idx = i // cols, i % cols

        if transform == "normal":
            clean_data = df[col].dropna()
            
        elif transform == "log":
            clean_data = np.log10(df[col].dropna() + 1) # + 1 for zero value (not neg)
            axes[row_idx, col_idx].xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"$10^{{{int(x)}}}$")
            )
            
        else:
            nah_not_implement()

        axes[row_idx, col_idx].set_xlabel(col)
        axes[row_idx, col_idx].set_title(f"Distribution of {col}")

        if plot == "histogram":
            axes[row_idx, col_idx].hist(clean_data, bins=30)
        elif plot == "boxplot":
            sns.boxplot(x=clean_data, ax=axes[row_idx, col_idx])
        else:
            nah_not_implement()
    
    plt.tight_layout()
    plt.show()

    return


################ End of Numeric ####################

################ Categorical #######################

def categorical_stat_report(df: pd.DataFrame):
    categorical_feature = list(df.select_dtypes(include=["object", "category"]).columns)
    for col in categorical_feature:
        print(f"================== Top 10 {col} ==================")
        print("Unique value:", df[col].nunique())
        print("Missing value:", df[col].isna().sum())
        display(df[col].value_counts().head(10))
    
    return categorical_feature

def categorical_plot(df:pd.DataFrame, config, plot):
    if config is None:
        config = {
            "normal": list(df.select_dtypes(include=["object", "category"]).columns)
        }

    categorical_feature = [(top_k, col) for top_k, columns in config.items() for col in columns]
    rows = len(categorical_feature)
    print(categorical_feature)
    
    for it, (top_k, col) in enumerate(categorical_feature):
        label_counts = df[col].value_counts()
        if top_k == "top_10":
            label_counts = label_counts[:10]
        elif top_k != "normal":
            print("Unknown top_k")
            raise ValueError
        
        plt.figure(figsize=(len(label_counts) // 2 + 1, 5))
        plt.xlabel(col)
        plt.xticks(rotation=45, ha="right")

        if plot == "barplot":
            sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
            plt.title(f"Bar plot of {col}")
        # elif plot == "pieplot":
        #     plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
        #     plt.title(f"Pie plot of {col}")
        else:
            print("Unknown plot type")
            raise ValueError

        for i, v in enumerate(label_counts.values):
            plt.text(i, v + 5, str(v), ha="center", va="bottom")
        
        plt.tight_layout()
        plt.show()

############### End of Categorical #################

################################# End of Descriptive Statistics #######################################



################################# Preprocessing #######################################

def make_column_pipeline(config: dict) -> Pipeline:
    """
    config: dict {"name": (impute, [list_of_columns_index])}. Default = None = dropna
    Example:
    config = {
        "impute": (SimpleImputer(strategy="mean"), [0, 1])
    }
    """
    transformer = [(name, impute, cols) for name, (impute, cols) in config.items()]
    return Pipeline(steps=[
        ("missing_value", ColumnTransformer(transformers=transformer, remainder="passthrough"))
    ])

def make_preprocess_pipeline(step_list: list):
    return Pipeline(steps=[(f"step_{i}",step) for i, step in enumerate(step_list)])

def get_preprocesser(step: str, type: str):

    num_imputer_dict = {
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "constant": SimpleImputer(strategy="constant")
    }

    cate_imputer_dict = {
        "most": SimpleImputer(strategy="most_frequent"),
        "constant": SimpleImputer(strategy="constant")
    }

    scaler_dict = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "log1p_robust": Log1pRobustScaler()
    }

    pca_dict = {
        "pca_0.95_auto": PCA(n_components=0.95, svd_solver="auto"),
        "pca_0.95_full": PCA(n_components=0.99, svd_solver="full"),
        "pca_0.99_auto": PCA(n_components=0.99, svd_solver="auto"),
        "pca_0.99_full": PCA(n_components=0.99, svd_solver="full")
    }

    encoder_dict = {
        "onehot": OneHotEncoder(handle_unknown='ignore', sparse_output=False),
        "ordinal": OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    }

    step_dict = {
        "num_impute": num_imputer_dict,
        "cate_impute": cate_imputer_dict,
        "scale": scaler_dict,
        "pca": pca_dict,
        "encode": encoder_dict
    }

    if step in step_dict:
        if type == "all":
            return step_dict[step]
        elif type in step_dict[step]:
            return step_dict[step][type]
        else:
            print(f"Unknown {type} of step {step}")
            raise ValueError
    else:
        print("Unknown step")
        raise ValueError


################################# End of Preprocessing #######################################

def multi_preprocess(preprocess_config: list, data: pd.DataFrame, target: str):
    results = pd.DataFrame(columns=[name for name in preprocess_config[0]] + ["data"])

    data = data.dropna(subset=[target])

    numeric_cols = list(data.select_dtypes(include=["number"]).columns)
    cate_cols = list(data.select_dtypes(include=["object", "category"]).columns)

    if target in numeric_cols:
        numeric_cols.remove(target)
    else:
        cate_cols.remove(target)

    for config in preprocess_config:

        if config["num_impute"] == "none":
            data = data.dropna(subset=numeric_cols)
        if config["cate_impute"] == "none":
            data = data.dropna(subset=cate_cols)

        X = data.drop(columns=target)
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        numeric_steps = [(step, get_preprocesser(step, config[step])) for step in ("num_impute", "scale", "pca") if config[step] != "none"]
        cate_steps = [(step, get_preprocesser(step, config[step])) for step in ("cate_impute", "encode") if config[step] != "none"]
        if len(numeric_steps) == 0 and len(cate_steps) == 0:
            continue

        transformers = []
        if len(numeric_steps) != 0:
            transformers.append(("numeric", Pipeline(steps=numeric_steps), numeric_cols))
        if len(cate_steps) != 0:
            transformers.append(("category", Pipeline(steps=cate_steps), cate_cols))
        
        pipe = ColumnTransformer(transformers=transformers, remainder="passthrough")
        # numeric_pipe = Pipeline(steps=numeric_steps)
        # cate_pipe = Pipeline(steps=cate_steps)
        # pipe = ColumnTransformer(transformers=[
        #     ("numeric", numeric_pipe, numeric_cols),
        #     ("category", cate_pipe, cate_cols)
        # ])

        X_train = pipe.fit_transform(X_train)
        X_test = pipe.transform(X_test)

        results.loc[len(results)] = [config[step] for step in config] + [(X_train, X_test, y_train, y_test)]
    
    return results

################################# Trainning #######################################

def train_one_model(params, split_datasets, score, y_transform=None):
    results = split_datasets.drop(columns=["data"])
    results[score] = 0
    for i, (X_train, X_test, y_train, y_test) in tqdm(
        enumerate(split_datasets["data"]), total=len(split_datasets), desc="Training"
    ):
        model = Pipeline(steps=[("model", params["model"])])
        model.set_params(**params)
        model.set_output(transform="pandas")

        if y_transform is not None:
            y_train = y_transform.fit_transform(y_train.values.reshape(-1, 1))
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        
        if score == "r2":
            if y_transform is not None:
                y_pred = y_transform.inverse_transform(y_pred.reshape(-1, 1))

            results[score][i] = r2_score(y_test, y_pred)

        elif score == "f1":
            if y_transform is not None:
                y_test = y_transform.transform(y_test.values.reshape(-1, 1))
            
            results[score][i] = f1_score(y_test, y_pred)
        else:
            raise ValueError
        
    return results

################################# End of Trainning #######################################

# ...existing code...

def train_multi_model(grid, params_df, dataset, score_name="r2", y_scaler=None):
    """
    Huấn luyện nhiều cấu hình siêu tham số (Grid Search thủ công) trên một bộ dữ liệu cụ thể.
    
    Args:
        grid (list): Danh sách các dict tham số (từ ParameterGrid).
        params_df (pd.DataFrame): DataFrame chứa các tham số (để lưu kết quả).
        data_path (str): Đường dẫn đến thư mục chứa dữ liệu (X_train.csv, y_train.csv, ...).
        score_name (str): Tên metric chính để sort kết quả.
        y_scaler (object): Scaler dùng để inverse_transform y (nếu có).
    
    Returns:
        pd.DataFrame: DataFrame kết quả bao gồm tham số và metrics.
    """

    # 1. Load dữ liệu từ đường dẫn
    # try:
    #     X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
    #     y_train = pd.read_csv(os.path.join(data_path, "y_train.csv"))
    #     X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
    #     y_test = pd.read_csv(os.path.join(data_path, "y_test.csv"))
    # except FileNotFoundError:
    #     print(f"Error: Data files not found in {data_path}")
    #     return pd.DataFrame()

    (X_train, X_test, y_train, y_test) = dataset

    if y_scaler is not None:
        y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

    results = []

    # 2. Lặp qua từng bộ tham số
    for params in tqdm(grid, desc="Training Multi Models"):
        model = params['model']
        
        # Lọc bỏ key 'model' và xóa tiền tố 'model__' nếu có để set_params
        clean_params = {k.replace("model__", ""): v for k, v in params.items() if k != 'model'}
        model.set_params(**clean_params)

        # Huấn luyện
        # y_train.values.ravel() để chuyển thành mảng 1 chiều nếu cần
        model.fit(X_train, y_train)
        
        # Dự đoán
        y_pred_scaled = model.predict(X_test)

        # Inverse transform nếu có scaler (để tính metric trên giá trị thực)
        if y_scaler:
            # Reshape (-1, 1) vì scaler yêu cầu mảng 2D
            y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
            # Lưu ý: y_test load từ file csv thường là giá trị gốc (chưa scale) 
            # hoặc đã scale tùy vào cách bạn lưu ở bước preprocess.
            # Ở đây giả định y_test từ file là Ground Truth (giá trị thực).
        else:
            y_pred = y_pred_scaled

        # Tính toán metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })

    # 3. Tổng hợp kết quả
    results_df = pd.DataFrame(results)
    # Ghép cột tham số và cột kết quả
    final_df = pd.concat([params_df.reset_index(drop=True), results_df], axis=1)
    
    return final_df.sort_values(by=score_name, ascending=(score_name != "r2")) 
    # Nếu score là r2 thì giảm dần (cao là tốt), ngược lại (mse, rmse) thì tăng dần (thấp là tốt)





from sklearn.base import BaseEstimator, TransformerMixin

class Log1pRobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = RobustScaler()
        self.columns_ = None
        self.index_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
            self.index_ = X.index
        X_log = np.log1p(X)
        self.scaler.fit(X_log)
        return self

    def transform(self, X):
        X_log = np.log1p(X)
        X_scaled = self.scaler.transform(X_log)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_scaled, columns=self.columns_, index=X.index)
        return X_scaled

    def inverse_transform(self, X):
        X_inv = self.scaler.inverse_transform(X)
        X_orig = np.expm1(X_inv)
        if self.columns_ is not None:
            return pd.DataFrame(X_orig, columns=self.columns_)
        return X_orig

    def set_output(self, *, transform = None): # Dummy function
        return self

def nah_not_implement():
    print("Nah, update later :)")
    raise NotImplementedError