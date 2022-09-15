# DADS7202 - HW1

## Highlight
สรุปไฮไลท์ที่น่าสนใจของการบ้านชิ้นนี้ โดยเขียนเป็น bullet สั้น ๆ กระชับแต่ได้ใจความ จํานวนประมาณ 3-5 bullets ทั้งนี้ให้ใส่ส่วนนี้ไว้ที่ด้านบนสุดของหน้าเพจ (หลังจาก topic) โดยควรจะเป็นข้อคิดเห็น การค้นพบ ข้อสรุปหรือข้อมูล insight น่าสนใจที่ทางกลุ่มค้นพบจากการทําการบ้านครั้งนี้

## Data
[Churn Modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling): คำอธิบายของ Data
```
url = "https://drive.google.com/file/d/1-mT6iykRVgRU3blYpX5i_YxhbqjUJepP/view?usp=sharing"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path)
```
**Data visualization**

```
df.head()
```
<img width="900" alt="image" src="https://user-images.githubusercontent.com/97492504/190356749-70acd818-e305-4c69-85a5-67b1469d2da5.png">

```
df.describe()
```
<img width="940" alt="image" src="https://user-images.githubusercontent.com/97492504/190357180-78e37f5d-38b7-4807-b2d8-191c8540b980.png">


```
fig, ax = plt.subplots(2,3,figsize=(12,10))
fig.tight_layout(pad=2, w_pad=5 , h_pad=2)
sns.countplot(df['Exited'], ax=ax[0,0])
ax[0,0].set_xticklabels(["Retained", "Closed"])
sns.countplot(df['Gender'], ax=ax[0,1])
sns.countplot(df['Geography'], ax=ax[0,2])
sns.countplot(df['HasCrCard'], ax=ax[1,0])
ax[1,0].set_xticklabels(["No", "Yes"])
sns.countplot(df['IsActiveMember'], ax=ax[1,1])
ax[1,1].set_xticklabels(["No", "Yes"])
sns.countplot(df['NumOfProducts'], ax=ax[1,2])
```
<img width="600" alt="image" src="https://user-images.githubusercontent.com/97492504/190357564-9eb10355-1a0b-4d54-a26b-b53a5d9bc85e.png">

```
plt.figure(figsize=(12, 9))
sns.histplot(df['Age'],color="green",fill=False)
```
<img width="300" alt="image" src="https://user-images.githubusercontent.com/97492504/190357858-5b59a9a3-79ac-4f28-8a92-a185fe62a885.png">

```
plt.figure(figsize=(12, 9))
sns.histplot(df['Balance'],color="orange",fill=False)
```
<img width="300" alt="image" src="https://user-images.githubusercontent.com/97492504/190358043-a63e29fc-db3f-41d7-89fe-f527d345b27c.png">

```
plt.figure(figsize=(12, 9))
sns.histplot(df['EstimatedSalary'],color="purple",fill=False)
```
<img width="300" alt="image" src="https://user-images.githubusercontent.com/97492504/190358273-e9926fe6-ee28-4286-8162-7682a91fb166.png">

## Exploratory Data Analysis : EDA
**Data preparation and Data pre-processing**: Data Cleaning, Normalization and Handling Imbalanced Data.
```
df.drop(columns=['RowNumber', 'CustomerId','Surname'], axis=1, inplace=True)
```

```
df['Gender'].replace(['Female'],0 , inplace=True)
```

```
df['Gender'].replace(['Male'],1 , inplace=True)
```

```
df['Gender'].unique()
```

```
dummy=pd.get_dummies(df.Geography)
df=pd.concat([df, dummy], axis=1)
df.drop('Geography', axis=1, inplace=True)
```

```
y = df['Exited']
df.drop(columns=['Exited'], axis=1, inplace=True)
```

```
df
```
<img width="800" alt="image" src="https://user-images.githubusercontent.com/97492504/190359905-9ffa8d58-52ed-4a58-b411-00b776ea3b52.png">

```
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
scalar = preprocessing.MinMaxScaler()
col=df.columns
df=scalar.fit_transform(df)
df=pd.DataFrame(df,columns=col)
```

```
df=pd.DataFrame(df,columns=col)
```

```
df
```
<img width="800" alt="image" src="https://user-images.githubusercontent.com/97492504/190361206-a28fbb68-7b2d-426b-8e7e-929949b1c7ab.png">

```
from imblearn.over_sampling import SMOTE
smote=SMOTE(sampling_strategy='auto',k_neighbors=1,random_state=1234)
X_data, Y_data = smote.fit_resample(df,y)
```

```
X_data
```
<img width="800" alt="image" src="https://user-images.githubusercontent.com/97492504/190361752-aeed69ad-b52e-491b-8b8f-1da742c0c543.png">

```
sns.countplot(Y_data)
```
<img width="300" alt="image" src="https://user-images.githubusercontent.com/97492504/190362007-c3d4b156-cc6b-45f6-a391-cd983ba7ead8.png">

## Machine Learning Model : ML

```
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_data , Y_data , test_size = 0.3, random_state = 0)
```

```
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
Y_train = Y_train.astype(np.float32)
Y_test = Y_test.astype(np.float32)
```

```
from sklearn import metrics
from sklearn.metrics import classification_report , accuracy_score , roc_curve , roc_auc_score , confusion_matrix
```

```
def predict_y(X_test,model):
  if str(type(model)) == "<class 'keras.engine.sequential.Sequential'>":
    Y_predict = model.predict(X_test)
    for i in range(0, len(Y_predict)):
      if Y_predict[i] > 0.5:
        Y_predict[i] = 1
      else:
        Y_predict[i] = 0
  else:
    Y_predict=model.predict(X_test)
  return Y_predict
```
 
```
def acc_score(X_test,Y_test,model):
  Y_predict = predict_y(X_test,model)
  acc = accuracy_score(Y_test, Y_predict)
  print(f"Accuracy = {acc*100:.2f}%")

def class_report (X_test,Y_test,model):
  Y_predict = predict_y(X_test,model)
  print(f'\nClassification Report\n')
  print(classification_report(Y_test,Y_predict))

def plot_crv(X_test,Y_test,model):
  Prob_Y_predict = model.predict_proba(X_test)[::,1]
  print(f"\nAUC_ROC = {roc_auc_score(Y_test,Prob_Y_predict)*100:.2f}%\n")
  fpr, tpr, _ = metrics.roc_curve(Y_test,  Prob_Y_predict)
  plt.plot(fpr,tpr)
  plt.plot([0,1], [0,1], 'red')
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.show()

def plot_cfm(X_test,Y_test,model):
  if str(type(model)) == "<class 'keras.engine.sequential.Sequential'>":
    shd='Purples'
  else:
    shd='YlGnBu'
  Y_predict = predict_y(X_test,model)
  cf_matrix = confusion_matrix(Y_test, Y_predict)
  print("\nConfusion Matrix")
  print(cf_matrix)
  ax = sns.heatmap(cf_matrix, annot=True, cmap=shd,fmt='g')

  ax.set_title('Confusion Matrix Heatmap');
  ax.set_xlabel('Predicted Values')
  ax.set_ylabel('Actual Values ');

  ax.xaxis.set_ticklabels(['Retrained','Closed'])
  ax.yaxis.set_ticklabels(['Retrained','Closed'])
  plt.show()
```

**Decision Tree in ML**
```
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(X_train, Y_train)
acc_score(X_test,Y_test,DT)
class_report (X_test,Y_test,DT)
plot_crv(X_test,Y_test,DT)
plot_cfm(X_test,Y_test,DT)
```
<img width="340" alt="image" src="https://user-images.githubusercontent.com/97492504/190375560-a828890d-84b6-421d-9f94-f40512feab61.png">    <img width="312" alt="image" src="https://user-images.githubusercontent.com/97492504/190365062-c7e3f941-9d6a-449e-b6f0-4ea3b35d7945.png">    <img width="273" alt="image" src="https://user-images.githubusercontent.com/97492504/190365126-3e6c6760-099b-4c9d-8bf9-eb509223925f.png">

**Random Forest in ML**
```
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(X_train, Y_train)
acc_score(X_test,Y_test,RF)
class_report (X_test,Y_test,RF)
plot_crv(X_test,Y_test,RF)
plot_cfm(X_test,Y_test,RF)
```
<img width="340" alt="image" src="https://user-images.githubusercontent.com/97492504/190377521-2575e1bf-4e0b-4386-b09d-7c42891a6d9c.png">     <img width="312" alt="image" src="https://user-images.githubusercontent.com/97492504/190376822-9aefe40e-ea3c-440c-ad48-d2eea11bcd72.png">     <img width="273" alt="image" src="https://user-images.githubusercontent.com/97492504/190376957-1a3f7056-36de-499f-a793-04ce484208e2.png">

