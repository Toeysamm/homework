# DADS7202 - HW1  

## Introduction: 
Customer churn is a major problem and one of the most important concerns for large companies. Due to the direct effect on the revenues of the companies, especially in the Bank. Therefore this project doing churn prediction by using  data set which  includes details of a bank's customers and the target variable is a binary variable reflecting the fact whether the customer left the bank (closed his account) or  continues to be a customer.  

For  this project will do  binary classification by using Traditional ML and MLP.


## Data
[Churn Modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling):  
The dataset contains 10,000 rows and 14 columns  
 1. RowNumber: Row Numbers from 1 to 10000
 2. CustomerId: Unique Ids for bank customer identification
 3. Surname: Customer's last name
 4. CreditScore: Credit score of the customer
 5. Geography: The country from which the customer belongs
 6. Gender: Male or Female
 7. Age: Age of the customer 
 8. Tenure: Number of years for which the customer has been with the bank
 9. Balance: Bank balance of the customer
 10. NumOfProducts: Number of bank products the customer is utilising
 11. HasCrCard : Binary Flag for whether the customer holds a credit card with the bank or not
 12. IsActiveMember : Binary Flag for whether the customer is an active member with the bank or not
 13. EstimatedSalary : Estimated salary of the customer in Dollars
 14. Exited : Binary flag 1 if the customer closed account with bank and 0 if the customer is retained  

Column ‘Exited’ is target variable (Y_train, Y_test)


```
url = "https://drive.google.com/file/d/1-mT6iykRVgRU3blYpX5i_YxhbqjUJepP/view?usp=sharing"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path)
```
**Exploring data with data visualisation**

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

Select some columns and drop useless columns from the dataframe, then we change parameters in Gender column to dummy variables.
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
For Geography column, we replace with three dummy data type columns.
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

Use MinMaxScaler to scaling the data.
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

From data visualisation above.
<center><img width="250" alt="image" src="https://user-images.githubusercontent.com/97492504/190437992-ba449b0f-9f03-4627-bbb1-c61d964878c9.png"></center>  
As we see in the data visualisation , in column ‘Exited ‘ which is imbalance data so we use resampling to adjust it into balance data.

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

**Data formatting**
```
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
Y_train = Y_train.astype(np.float32)
Y_test = Y_test.astype(np.float32)
print(f"X_train.shape={X_train.shape}")
print(f"Y_train.shape={Y_train.shape}")
print(f"X_test.shape={X_test.shape}")
print(f"Y_test.shape={Y_test.shape}")
```
<img width="200" alt="image" src="https://user-images.githubusercontent.com/97492504/190440384-ad913f9d-397f-472e-ad29-d55f92a4d3fb.png">  
Format the data for processing in MLP Model  

 - Change data type to folat32  
  
### **Prepare the environment**  
GPU 0: Tesla T4 (UUID: GPU-6920f9b5-6b3f-b622-1489-c27afa2cf745)  

**Initial random weights**
```
np.random.seed(1150)
tf.random.set_seed(1112)
```

### **Create Network architecture**

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
mlp_model.add( tf.keras.layers.Dense(32, activation='relu', name='hidden1') )  
mlp_model.add( tf.keras.layers.Dense(16, activation='relu', name='hidden2') )   
mlp_model.add( tf.keras.layers.Dense(8, activation='relu', name='hidden3') )   
mlp_model.add( tf.keras.layers.Dense(4, activation='relu', name='hidden4') )   
mlp_model.add( tf.keras.layers.Dense(2, activation='relu', name='hidden5') )   

# Output layer
mlp_model.add(Dense(1, activation = 'sigmoid'))


mlp_model.summary()
```

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/190415003-76468d59-30dc-4fbb-ac0d-0cc97a77433a.png">  
We didn’t use Batch Normalization (BN) layer because when using it the output from dense layer will stay within a fixed range.  


### **Compile the model**  
```
mlp_model.compile(optimizer='adam', loss = 'binary_crossentropy',metrics=['acc'])
#model.compile( optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'] )
```
 - **Optimizer : Adam**  
Use Adam as an Optimizer in this model.   
Cause adam has  qualities of Momentum and RMSprop that is based on adaptive estimation of first-order and second-order moments.  

 - **Loss function : binary_crossentropy**  
Binary_crossentropy is the most suitable Loss function for doing binary classification model.    
  
  
  
#### **As selecting the best number of epoch, we need help.**  
Therefore we using earlystopping which stop at the best point and plus 20 epoch and ModelCheckpoint to find the maximum accuracy.
```
checkpoint_filepath = "bestmodel_epoch{epoch:02d}_valloss{val_loss:.2f}.hdf5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_filepath,
                                                                                              save_weights_only=True,
                                                                                              monitor='val_acc',
                                                                                              mode='max',
                                                                                              save_best_only=True)

from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 20, 
                                        restore_best_weights = True)
```

```
history = mlp_model.fit ( X_train, Y_train, batch_size=128, epochs=300, verbose=1, validation_split=0.2, callbacks=[model_checkpoint_callback,earlystopping] )
```

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

After we ran the MLP model several times with difference hyperparameters, finally we found the satisfactory model by using following these hyperparameters.  
 - Epochs = 300 
 - Batch size = 128
 - validation_split = 0.2  

<img width="800" alt="image" src="https://user-images.githubusercontent.com/97492504/190416192-42549e48-178f-4c65-a783-81e77163f6ce.png">

```
results = mlp_model.evaluate(X_test, Y_test, batch_size=128)
print( f"{mlp_model.metrics_names} = {results}" )
```
<img width="600" alt="image" src="https://user-images.githubusercontent.com/97492504/190461513-50842255-c28d-49d6-868c-f80029179428.png">

```
acc_score(X_test,Y_test,mlp_model)
class_report (X_test,Y_test,mlp_model)
plot_cfm(X_test,Y_test,mlp_model)
```
<img width="380" alt="image" src="https://user-images.githubusercontent.com/97492504/190461096-4117bba8-0b26-4bbf-b3ab-55d1c40666e8.png"><img width="320" alt="image" src="https://user-images.githubusercontent.com/97492504/190461239-bed46e28-4ebd-43b0-849b-217d0399a91b.png">    

Experiment with different initial random weights for 5 rounds  
Remark: Epoch numbers besd on Earlystopping  

MLP | Initial random weight | Epoch | Time (mean) | Time (SD) | Accuracy
------------ | ------------- | ------------- | ------------- | ------------- | -------------
Round 1 | np.random.seed(1150)  tf.random.set_seed(1112) | 173 | 0.039 | 0.322 | 0.8020 
Round 2 | np.random.seed(1234)  tf.random.set_seed(5678) | 188 | 0.106 | 0.353 | 0.8049 
Round 3 | np.random.seed(1928)  tf.random.set_seed(3746) | 78 | 0.210 | 0.495 | 0.7869 
Round 4 | np.random.seed(4321)  tf.random.set_seed(8765) | 82 | 0.225 | 0.499 | 0.7991 
Round 5 | np.random.seed(1324)  tf.random.set_seed(5769) | 131 | 0.158 | 0.422 | 0.8083 



## Discussion :
 1. Assumptions, this project is a comparison of the model between Traditional ML Model and MLP Model. This project is a Churn model by Binary Classification. Therefore, we considers that using MLP Model will be a better model than ML Model, the accuracy score will be more and the running time should be more because the model is larger and more complex than ML Model. But after running these model, we found that accuracy score of the traditional ML better than MLP, while using more time than others except Decision Tree.  
 2. Imbalance Data: we use resampling helps to balance the data. After resampling and running the model, it will increase the accuracy of ML Model but slightly decrease in MLP Model.  


## Conclusion: 
Based on the traditional ML, we selected four algorithms namely Decision Tree, Random Forest, Gradient Boosting and XGBoost, by estimating the model from Accuracy and running time to compare with MLP as shown in the table.  

:bellhop_bell: | Time | Accuracy
------------ | ------------- | -------------
MLP | 0.147±0.077 | 0.8003±0.0082
Decision Tree | 0.02 | 0.8309
Random Forest | 0.3 | 0.9004
Gradient Boosting | 0.4 | 0.8472 
XGBoost | 0.3 | 0.8753

The results obtained from the Accuracy of both traditional ML and MLP differ slightly. Overall, Traditional ML has a higher Accuracy, possibly due to less complex or low-volume data, but MLP run times are relatively faster than Traditional ML. 

## Member:  
6410412002  Mr. Kittipat Pattarajariya  
6410412004  Miss Chonthanya Yosbuth  
6410412010  Mr. Saran Ditjarern  
6410412018  Miss Panumas sitthikarn  


## Reference:  
Brownlee, J. (2020).   Random Oversampling and Undersampling for Imbalanced Classification. Retrieved from https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/  
Rahul, A. (2019). The 5 Sampling Algorithms every Data Scientist need to know. Retrieved from https://towardsdatascience.com/the-5-sampling-algorithms-every-data-scientist-need-to-know-43c7bc11d17c  
Ekarat, R. (2020). DADS6003 In-class competition Applied ML competition. Retrieved from https://www.kaggle.com/competitions/dads6003-in-class-competition/rules  
Shruti, I. (2019). Churn Modelling classification data set. Retrieved from https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling  


## End Credit:  
This work is part of the DADS7202 Deep Learning in Master degree of Science at Faculty of Applied Statistics National Institute of Development Administration.
