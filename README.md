# Dcard 文章愛心數預測演算法
這個Project的目的是，在給予文章發佈後前6小時的狀態及基本資訊，預測 24 小時後的愛心數。將使用機器學習的方法做回歸預測。

**文章狀態與基本資訊**
* `title`: 文章標題
* `created_at`: 文章發佈時間
* `like_count_1~6h`: 文章發佈後 1~6 小時的累積愛心數
* `comment_count_1~6h`: 文章發佈後 1~6 小時的累積留言數
* `forum_id`: 文章發佈看板 ID
* `author_id`: 文章作者 ID
* `forum_stats`: 看板資訊
## 環境需求
* 安裝Python3的虛擬環境。
* 安裝Pytorch後端框架，對於該如何在你使用的平台上安裝這些框架，可以參考PyTorch[安裝頁面](https://pytorch.org/get-started/locally/#start-locally)。
* 在虛擬環境下安裝🤗Transformers。
``` python
pip install transformers
```
## 方法
這個Project使用了4種機器學習演算法，分別是：
* Linear Regression in `ML.py`
* Logistic Regression in `ML.py`
* Support Vector Regression (SVR) in `ML.py`
* NLP Regression Model With 🤗Transformers in `NLP.py`
## 資料前處理
將需要的資料下載下來並存為`train`，`valid`和`test`，資料分別是5000筆，10000筆及10000筆。
``` python
import pandas as pd
train = pd.read_csv("./intern_homework_train_dataset.csv")
valid = pd.read_csv("./intern_homework_public_test_dataset.csv")
test = pd.read_csv("./intern_homework_private_test_dataset.csv")
```
檢查資料是否有空值，發現3個資料都沒有空值。
``` python
print("train data\n", train.isnull().sum())
print("validation data\n",valid.isnull().sum())
print("test data\n", test.isnull().sum())
```
* train data

|colums     |number of null|
|-----------|:------------:|
|title           |0|
|created_at      |0|
|like_count_1h   |0|
|like_count_2h   |0|
|like_count_3h   |0|
|like_count_4h   |0|
|like_count_5h   |0|
|like_count_6h   |0|
|comment_count_1h|0|
|comment_count_2h|0|
|comment_count_3h|0|
|comment_count_4h|0|
|comment_count_5h|0|
|comment_count_6h|0|
|forum_id        |0|
|author_id       |0|
|forum_stats     |0|
|like_count_24h  |0|
* validation data

|colums     |number of null|
|-----------|:------------:|
|title           |0|
|created_at      |0|
|like_count_1h   |0|
|like_count_2h   |0|
|like_count_3h   |0|
|like_count_4h   |0|
|like_count_5h   |0|
|like_count_6h   |0|
|comment_count_1h|0|
|comment_count_2h|0|
|comment_count_3h|0|
|comment_count_4h|0|
|comment_count_5h|0|
|comment_count_6h|0|
|forum_id        |0|
|author_id       |0|
|forum_stats     |0|
|like_count_24h  |0|
* test data

|colums     |number of null|
|-----------|:------------:|
|title           |0|
|created_at      |0|
|like_count_1h   |0|
|like_count_2h   |0|
|like_count_3h   |0|
|like_count_4h   |0|
|like_count_5h   |0|
|like_count_6h   |0|
|comment_count_1h|0|
|comment_count_2h|0|
|comment_count_3h|0|
|comment_count_4h|0|
|comment_count_5h|0|
|comment_count_6h|0|
|forum_id        |0|
|author_id       |0|
|forum_stats     |0|
|like_count_24h  |0|
接下來要檢查的是，data裡面不同種類的值有多少個，這邊用train data來舉例。
``` python
train.nunique()
```
實作後會發現基本上沒有特殊的某些類別。
|colums     |number of value|
|-----------|:------------:|
|title           |49158|
|created_at      |49654|
|like_count_1h   |112|
|like_count_2h   |185|
|like_count_3h   |250|
|like_count_4h   |308|
|like_count_5h   |380|
|like_count_6h   |438|
|comment_count_1h|172|
|comment_count_2h|216|
|comment_count_3h|252|
|comment_count_4h|277|
|comment_count_5h|305|
|comment_count_6h|331|
|forum_id        |1147|
|author_id       |32280|
|forum_stats     |221|
|like_count_24h  |940|
將目標的`like_count_24`與其他欄位的相關性印出來，發現相關性相對高的是`like_count_5h`和`like_count_6h`。
``` python
import matplotlib.pyplot as plt
import seaborn as sns

corrmat = train.corr()['like_count_24h']
corrmat
```
|colums     |correlation|
|-----------|:------------:|
|like_count_1h   |0.398345|
|like_count_2h   |0.467885|
|like_count_3h   |0.559062|
|like_count_4h   |0.648533|
|like_count_5h   |0.713661|
|like_count_6h   |0.760585|
|comment_count_1h|0.037055|
|comment_count_2h|0.048422|
|comment_count_3h|0.061782|
|comment_count_4h|0.074955|
|comment_count_5h|0.086810|
|comment_count_6h|0.094525|
|forum_id        |0.028790|
|author_id       |-0.004098|
|forum_stats     |0.045711|
|like_count_24h  |1.000000|
前處理的最後，剔除不要的欄位，做出要放進模型的data。
``` python
train.drop(columns={'title','created_at','forum_id','author_id','forum_stats'}, inplace=True)
valid.drop(columns={'title','created_at','forum_id','author_id','forum_stats'}, inplace=True)
```
### 演算法
在訓練各種模型之前，首先，先將資料整理成training和validation兩個部分。
``` python
import numpy as np
X_train = np.array(train.iloc[:,:-1])
X_valid = np.array(valid.iloc[:,:-1])
y_train = np.array(train['like_count_24h'].tolist())
y_valid = np.array(valid['like_count_24h'].tolist())
```
* Linear Regression
``` python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

model = LinearRegression()
model.fit(X_train , y_train)
#Predictions on Testing data
y_pred = model.predict(X_valid) 
# around() Evenly round to the given number of decimals.
r2 = r2_score(y_valid, np.around(y_pred,0))
mse = mean_squared_error(y_valid, np.around(y_pred,0))
mape = mean_absolute_percentage_error(y_valid, np.around(y_pred,0))*100

print('MSE:', mse)
print('R2 Score:', r2)
print('MAPE:', mape)
```
MSE: 13471.8647
R2 Score: 0.5194420923678873
MAPE: 87.97600954035927
* Logistic Regression`filename.py`
``` python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_valid = sc_X.transform(X_valid)
model_log = LogisticRegression()
model_log.fit(X_train , y_train)
# Predictions on Testing data
y_pred = model_log.predict(X_valid) 

r2 = r2_score(y_valid, np.around(y_pred,0))
mse = mean_squared_error(y_valid, np.around(y_pred,0))
mape = mean_absolute_percentage_error(y_valid, np.around(y_pred,0))*100

print('MSE:', mse)
print('R2 Score:', r2)
print('MAPE:', mape)
```
MSE: 24845.2405
R2 Score: 0.11373985300664236
MAPE: 40.81011815359801
* Support Vector Regression（SVR）`filename.py`
``` python
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
```
`SVR(kernel='rbf',C=1)`
``` python
# SVR C=1
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_valid = sc_X.transform(X_valid)
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
#Predictions on Testing data
y_pred = svr.predict(X_valid)
r2 = r2_score(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
mape = mean_absolute_percentage_error(y_valid, np.around(y_pred,0))*100

print('MSE:', mse)
print('R2 Score:', r2)
print('MAPE:', mape)
```
MSE: 23013.16228320222
R2 Score: 0.17909232603754466
MAPE: 36.62957492416708
`SVR(kernel='rbf',C=1e3)`
``` python
# SVR C=1e3
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_valid = sc_X.transform(X_valid)
svr = SVR(kernel='rbf',C=1e3)
svr.fit(X_train, y_train)
#Predictions on Testing data
y_pred = svr.predict(X_valid)
r2 = r2_score(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
mape = mean_absolute_percentage_error(y_valid, np.around(y_pred,0))*100

print('MSE:', mse)
print('R2 Score:', r2)
print('MAPE:', mape)
```
MSE: 15019.091487019512
R2 Score: 0.4642506185845696
MAPE: 34.02267396110232
* NLP Regression Model With 🤗Transformers`filename.py`
NLP Regression Model 使用的是Hugging face🤗的預訓練模型。首先，先`import`相關的library。
``` python
import numpy as np
import pandas as pd
import transformers
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
```
因為這是一個使用自然語言處裡做回歸預測的模型，所以我們只用`title`來當作我們的特徵。
``` python
train_set = train[['title', 'like_count_24h']]
train_set = train_set.rename({'like_count_24h': 'labels'}, axis=1)
valid_set = valid[['title', 'like_count_24h']]
valid_set = valid_set.rename({'like_count_24h': 'labels'}, axis=1)
```
將data轉換成`DatasetDict`
``` python
dataset = DatasetDict({"train": Dataset.from_pandas(train_set),
                      "valid": Dataset.from_pandas(valid_set)})
dataset
```
Output
```
DatasetDict({
train: Dataset({
        features: ['title', 'labels'],
        num_rows: 50000
    })
    valid: Dataset({
        features: ['title', 'labels'],
        num_rows: 10000
    })
})
```
把dataset準備好，就可以來建構模型了！我們用的Tokenizer和Model都是🤗的[bert-base-chinese](https://huggingface.co/bert-base-chinese#model-details)預訓練模型，這個模型已經針對中文進行了預訓練，詳情可以參考原始[BERT](https://arxiv.org/abs/1810.04805)論文。
``` pytohn 
BASE_MODEL = "bert-base-chinese"
LEARNING_RATE = 2e-5
# MAX_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 20

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# set num_labels=1 -> linear regression model
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
```
接著，將資料變成一個一個的Token，所以我們把資料的`title`丟到Tokenizer裡面。
值得注意的是，除了原本的`title`和`labels`，還有3個feature，`input_ids`，`token_type_ids`以及`attention_mask`，這3個則是模型在做Self-attention時需要的向量。
``` python
def tokenize_function(examples):
    return tokenizer(examples["title"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets
```
Output
```
DatasetDict({
    train: Dataset({
        features: ['title', 'labels', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 50000
    })
    valid: Dataset({
        features: ['title', 'labels', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 10000
    })
})
```
在回歸的問題中，我們是要預測一個連續值，需要衡量預測值與真實值之間距離的指標。所以，我們創建一個函數`compute_metrics_for_regression`，以在訓練數據時使用它。
``` python
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def compute_metrics_for_regression(eval_pred):
    predictions, labels = eval_pred
    labels = labels.reshape(-1, 1)
    
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    mape = mean_absolute_percentage_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    single_squared_errors = ((predictions - labels).flatten()**2).tolist()
    
    # Compute accuracy 
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)
    
    return {"mse": mse, "mae": mae,"mape": mape, "r2": r2, "accuracy": accuracy}
```
而訓練時我們需要損失函數來評估模型是否有收斂，這邊的損失函數使用的是Mean Square Error (MSE)。
``` python
import torch
class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0][:, 0]
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss
```
最後，使用Transformers內建的TrainingArguments和Trainer來訓練模型。
``` python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="test_trainer_medium",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
    weight_decay=0.01,
)

trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    compute_metrics=compute_metrics_for_regression,
)
trainer.train()
```
最好的結果是MAPE=134.7。
### 結果
在訓練完以上的模型後面，最好的結果是由`SVR(kernel='rbf',C=1e3)`訓練出的模型(MAPE= 34.02)。所以，把前面前處理好的`test`丟到模型裡面做預測，並產出最後的結果，存成`result.csv`，就可以繳交了！
``` python
test.drop(columns={'title','created_at','forum_id','author_id','forum_stats'}, inplace=True)
result_test = np.array(test)
result_pred = svr.predict(result_test)
result_pred = pd.DataFrame(np.around(result_pred,0).astype(int), columns=['like_count_24h'])
result_pred.to_csv("./result.csv") 
