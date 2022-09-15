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
<img width="434" alt="image" src="https://user-images.githubusercontent.com/97492504/190357858-5b59a9a3-79ac-4f28-8a92-a185fe62a885.png">

```
plt.figure(figsize=(12, 9))
sns.histplot(df['Balance'],color="orange",fill=False)
```
<img width="434" alt="image" src="https://user-images.githubusercontent.com/97492504/190358043-a63e29fc-db3f-41d7-89fe-f527d345b27c.png">

```
plt.figure(figsize=(12, 9))
sns.histplot(df['EstimatedSalary'],color="purple",fill=False)
```
<img width="434" alt="image" src="https://user-images.githubusercontent.com/97492504/190358273-e9926fe6-ee28-4286-8162-7682a91fb166.png">

## EDA
**Data preparation and Data pre-processing**: Data Cleaning and normalization.
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
