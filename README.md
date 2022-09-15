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

**Data formatting**
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

**Gradient Boosting in ML**
```
from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier()
GB.fit(X_train, Y_train)
acc_score(X_test,Y_test,GB)
class_report (X_test,Y_test,GB)
plot_crv(X_test,Y_test,GB)
plot_cfm(X_test,Y_test,GB)
```
<img width="340" alt="image" src="https://user-images.githubusercontent.com/97492504/190379243-7b22dddc-d32b-4c8b-897e-f0de517f05d3.png">      <img width="312" alt="image" src="https://user-images.githubusercontent.com/97492504/190379348-503d3b48-8fb4-42bc-99e5-6e1ea6c8714b.png">     <img width="273" alt="image" src="https://user-images.githubusercontent.com/97492504/190379420-827486b6-aead-45f5-a5a0-cf21ec7ffc3a.png">

**XGBoost in ML**
```
from xgboost import XGBClassifier
xgb_model = XGBClassifier(learning_rate = 0.1, n_estimators = 180, max_depth = 3)
xgb_model.fit(X_train, Y_train)
acc_score(X_test,Y_test,xgb_model)
class_report (X_test,Y_test,xgb_model)
plot_crv(X_test,Y_test,xgb_model)
plot_cfm(X_test,Y_test,xgb_model)
```
<img width="340" alt="image" src="https://user-images.githubusercontent.com/97492504/190380227-f0823411-65ea-4aee-8979-85dd3e730185.png">     <img width="312" alt="image" src="https://user-images.githubusercontent.com/97492504/190380314-5a6bcd1a-cae1-4152-b216-49547c6d4053.png">      <img width="273" alt="image" src="https://user-images.githubusercontent.com/97492504/190380436-ea1f2a2f-39b0-4287-a228-37b861c36015.png">

## Multilayer perceptron : MLP

**set seed**  
`iteration 1`
```
np.random.seed(1150)
tf.random.set_seed(1112)
```
**coding for MLP**

```
X_train.shape
```
```
Y_train.shape
```
```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
```
```
mlp_model = tf.keras.models.Sequential()

# Input layer
mlp_model.add( tf.keras.Input(shape=(12,)) )

# Hidden layer
mlp_model.add( tf.keras.layers.Dense(20, activation='relu', name='hidden1') )  
mlp_model.add( tf.keras.layers.BatchNormalization(axis=-1, name='bn1') )  
mlp_model.add( tf.keras.layers.Dense(20, activation='relu', name='hidden2') )   
mlp_model.add( tf.keras.layers.BatchNormalization(axis=-1, name='bn2') )
mlp_model.add( tf.keras.layers.Dense(20, activation='relu', name='hidden3') )   
mlp_model.add( tf.keras.layers.BatchNormalization(axis=-1, name='bn3') )
mlp_model.add( tf.keras.layers.Dense(20, activation='relu', name='hidden4') )   
mlp_model.add( tf.keras.layers.Dropout(0.3) )                     

# Output layer
mlp_model.add(Dense(1, activation = 'sigmoid'))


mlp_model.summary()
```
<img width="400" alt="image" src="https://user-images.githubusercontent.com/97492504/190387698-f417a166-e441-46f5-beef-4df3ff545715.png">

```
mlp_model.compile(optimizer='adam', loss = 'binary_crossentropy',metrics=['acc'])
#model.compile( optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'] )
```
```
checkpoint_filepath = "bestmodel_epoch{epoch:02d}_valloss{val_loss:.2f}.hdf5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_filepath,
                                                                                              save_weights_only=True,
                                                                                              monitor='val_acc',
                                                                                              mode='max',
                                                                                              save_best_only=True)
```
```
history = mlp_model.fit ( X_train, Y_train, batch_size=128, epochs=25, verbose=1, validation_split=0.2, callbacks=[model_checkpoint_callback] )
```
<img width="700" alt="image" src="https://user-images.githubusercontent.com/97492504/190388476-c094e005-4431-4c83-8fe1-7176de10da1b.png">

```
# Summarize history for accuracy
plt.figure(figsize=(15,5))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Train accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.grid()
plt.show()

# Summarize history for loss
plt.figure(figsize=(15,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Train loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.grid()
plt.show()
```

<img width="800" alt="image" src="https://user-images.githubusercontent.com/97492504/190389375-ccad43c1-e5a6-44d3-9823-2834f4c88241.png">

```
results = mlp_model.evaluate(X_test, Y_test, batch_size=128)
print( f"{mlp_model.metrics_names} = {results}" )
```
<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/190389812-09932581-f744-4855-96f3-65bfcf52a5d8.png">

```
acc_score(X_test,Y_test,mlp_model)
class_report (X_test,Y_test,mlp_model)
plot_cfm(X_test,Y_test,mlp_model)
```

<img width="340" alt="image" src="https://user-images.githubusercontent.com/97492504/190389992-bdc5addd-6bbf-4049-a7aa-51fade2c5667.png">      <img width="300" alt="image" src="https://user-images.githubusercontent.com/97492504/190390049-91f2d21d-330a-44b5-acb3-79de2e667654.png">

`iteration 2`
```
np.random.seed(1234)
tf.random.set_seed(5678)
```
<img width="700" alt="image" src="https://user-images.githubusercontent.com/97492504/190392279-8009fddb-b544-470f-be3b-fe0196b8bdc6.png">
<img width="800" alt="image" src="https://user-images.githubusercontent.com/97492504/190392401-8cf15b86-00f7-4ddf-98d6-6eb8e576302e.png">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/190392558-64ec1518-05c8-43a0-b8ca-7786deecc1ec.png">
<img width="340" alt="image" src="https://user-images.githubusercontent.com/97492504/190392830-912fce20-5b1d-4d05-a1f1-1ae5f90b1d9f.png"><img width="300" alt="image" src="https://user-images.githubusercontent.com/97492504/190392917-2d8c9509-bf66-4751-93e7-365d313e128c.png">
