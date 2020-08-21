#!/usr/bin/env python
# coding: utf-8

# # Content

# + Data Cleaning
# + Exploratory Visualization
# + Feature Engineering
# + Basic Modeling & Evaluation
# + Hyperparameters tuning
# + Ensemble Methods

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


# combine train and test set.
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
combined_data=pd.concat([train,test],ignore_index=True)


# In[8]:


combined_data.head(5)


# ## First Let's take a look at our data graphically.

# In[155]:


# specifies the parameters of our graphs
fig = plt.figure(figsize=(18,6), dpi=1600) 
alpha=alpha_scatterplot = 0.2 
alpha_bar_chart = 0.55

# lets us plot many diffrent shaped graphs together 
ax1 = plt.subplot2grid((2,3),(0,0))
# plots a bar graph of those who surived vs those who did not.               
combined_data.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
# this nicely sets the margins in matplotlib to deal with a recent bug 1.3.1
ax1.set_xlim(-1, 2)
# puts a title on our graph
plt.title("Distribution of Survival, (1 = Survived)")    

plt.subplot2grid((2,3),(0,1))
plt.scatter(combined_data.Survived, combined_data.Age, alpha=alpha_scatterplot)
# sets the y axis lable
plt.ylabel("Age")
# formats the grid line style of our graphs                          
plt.grid(b=True, which='major', axis='y')  
plt.title("Survival by Age,  (1 = Survived)")

ax3 = plt.subplot2grid((2,3),(0,2))
combined_data.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)
ax3.set_ylim(-1, len(combined_data.Pclass.value_counts()))
plt.title("Class Distribution")

plt.subplot2grid((2,3),(1,0), colspan=2)
# plots a kernel density estimate of the subset of the 1st class passangers's age
combined_data.Age[combined_data.Pclass == 1].plot(kind='kde')    
combined_data.Age[combined_data.Pclass == 2].plot(kind='kde')
combined_data.Age[combined_data.Pclass == 3].plot(kind='kde')
 # plots an axis lable
plt.xlabel("Age")    
plt.title("Age Distribution within classes")
# sets our legend for our graph.
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 

ax5 = plt.subplot2grid((2,3),(1,2))
combined_data.Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
ax5.set_xlim(-1, len(combined_data.Embarked.value_counts()))
# specifies the parameters of our graphs
plt.title("Passengers per boarding location")


# # Data Cleaning

# In[9]:


combined_data.isnull().sum()


# __The 'Age', 'Cabin', 'Embarked', 'Fare' columns have missing values. First we fill the missing 'Embarked' with the mode.__

# In[10]:


combined_data.Embarked.mode()


# In[12]:


plt.subplot2grid((2,3),(0,1))
plt.scatter(df.Survived, df.Age, alpha=alpha_scatterplot)
# sets the y axis lable
plt.ylabel("Age")
# formats the grid line style of our graphs                          
plt.grid(b=True, which='major', axis='y')  
plt.title("Survival by Age,  (1 = Survived)")
['Embarked'].fillna('S',inplace=True)


# __Since 'Fare' is mainly related to 'Pclass', we should check which class this person belongs to.__

# In[13]:


combined_data[combined_data.Fare.isnull()]


# __It's a passenger from Pclass 3, so we'll fill the missing value with the median fare of Pclass 3.__

# In[14]:


combined_data.Fare.fillna(combined_data[combined_data.Pclass==3]['Fare'].median(),inplace=True)


# **There are a lot of missing values in 'Cabin', maybe there is difference between the survival rate of people who has Cabin number and those who hasn't.**

# In[15]:


combined_data.loc[combined_data.Cabin.notnull(),'Cabin']=1
combined_data.loc[combined_data.Cabin.isnull(),'Cabin']=0


# In[16]:


combined_data.Cabin.isnull().sum()


# In[17]:


pd.pivot_table(combined_data,index=['Cabin'],values=['Survived']).plot.bar(figsize=(8,5))
plt.title('Survival Rate')


# __We can also plot the count of 'Cabin' to see some patterns.__

# In[18]:


cabin=pd.crosstab(combined_data.Cabin,combined_data.Survived)
cabin.rename(index={0:'no cabin',1:'cabin'},columns={0.0:'Dead',1.0:'Survived'},inplace=True)
cabin


# In[19]:


cabin.plot.bar(figsize=(8,5))
plt.xticks(rotation=0,size='xx-large')
plt.title('Survived Count')
plt.xlabel('')
plt.legend()


# __From the plot, we can conclude that there is far more chance for a passenger to survive if we know his/her 'Cabin'.__

# ### Extract Title from 'Name'

# In[20]:


combined_data['Title']=combined_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())


# In[21]:


combined_data.Title.value_counts()


# In[22]:


pd.crosstab(combined_data.Title,combined_data.Sex)


# __All the 'Title' belongs to one kind of gender except for 'Dr'.__

# In[23]:


combined_data[(combined_data.Title=='Dr')&(combined_data.Sex=='female')]


# __So the PassengerId of the female 'Dr' is '797'. Then we map the 'Title'.__

# In[24]:


nn={'Capt':'Rareman', 'Col':'Rareman','Don':'Rareman','Dona':'Rarewoman',
    'Dr':'Rareman','Jonkheer':'Rareman','Lady':'Rarewoman','Major':'Rareman',
    'Master':'Master','Miss':'Miss','Mlle':'Rarewoman','Mme':'Rarewoman',
    'Mr':'Mr','Mrs':'Mrs','Ms':'Rarewoman','Rev':'Mr','Sir':'Rareman',
    'the Countess':'Rarewoman'}


# In[25]:


combined_data.Title=combined_data.Title.map(nn)


# In[26]:


# assign the female 'Dr' to 'Rarewoman'
combined_data.loc[combined_data.PassengerId==797,'Title']='Rarewoman'


# In[27]:


combined_data.Title.value_counts()


# In[28]:


combined_data[combined_data.Title=='Master']['Sex'].value_counts()


# In[29]:


combined_data[combined_data.Title=='Master']['Age'].describe()


# In[30]:


combined_data[combined_data.Title=='Miss']['Age'].describe()


# + __'Master' mainly stands for little boy, but we also want to find little girl. Because children tend to have higher survival rate.__

# + __For the 'Miss' with a Age record, we can simply determine whether a 'Miss' is a little girl by her age.__

# + __For the 'Miss' with no Age record, we use (Parch!=0). Since if it's a little girl, she was very likely to be accompanied by parents.__

# We'll create a function to filter the girls. The function can't be used if 'Age' is Nan, so first we fill the missing values with '999'.

# In[31]:


combined_data.Age.fillna(999,inplace=True)


# In[32]:


def girl(aa):
    if (aa.Age!=999)&(aa.Title=='Miss')&(aa.Age<=14):
        return 'Girl'
    elif (aa.Age==999)&(aa.Title=='Miss')&(aa.Parch!=0):
        return 'Girl'
    else:
        return aa.Title


# In[33]:


combined_data['Title']=combined_data.apply(girl,axis=1)


# In[34]:


combined_data.Title.value_counts()


# __Next we fill the missing 'Age' according to their 'Title'.__

# In[35]:


combined_data[combined_data.Age==999]['Age'].value_counts()


# In[36]:


Tit=['Mr','Miss','Mrs','Master','Girl','Rareman','Rarewoman']
for i in Tit:
    combined_data.loc[(combined_data.Age==999)&(combined_data.Title==i),'Age']=combined_data.loc[combined_data.Title==i,'Age'].median()


# In[37]:


combined_data.info()


# ### Finally, there is no missing value now!!!

# # Exploratory Visualization

# In[39]:


combined_data.head(5)


# __Let's first check whether the Age of each Title is reasonable.__

# In[40]:


combined_data.groupby(['Title'])[['Age','Title']].mean().plot(kind='bar',figsize=(8,5))
plt.xticks(rotation=0)
plt.show()


# __As we can see, female has a much larger survival rate than male.__

# In[41]:


pd.crosstab(combined_data.Sex,combined_data.Survived).plot.bar(stacked=True,figsize=(8,5),color=['#4169E1','#FF00FF'])
plt.xticks(rotation=0,size='large')
plt.legend(bbox_to_anchor=(0.55,0.9))


# __ We can also check the relationship between 'Age' and 'Survived'.__

# In[42]:


agehist=pd.concat([combined_data[combined_data.Survived==1]['Age'],combined_data[combined_data.Survived==0]['Age']],axis=1)
agehist.columns=['Survived','Dead']
agehist.head()


# In[43]:


agehist.plot(kind='hist',bins=30,figsize=(15,8),alpha=0.3)


# In[44]:


farehist=pd.concat([combined_data[combined_data.Survived==1]['Fare'],combined_data[combined_data.Survived==0]['Fare']],axis=1)
farehist.columns=['Survived','Dead']
farehist.head()


# In[45]:


farehist.plot.hist(bins=30,figsize=(15,8),alpha=0.3,stacked=True,color=['blue','red'])


# __People with high 'Fare' are more likely to survive, though most 'Fare' are under 100.__

# In[46]:


combined_data.groupby(['Title'])[['Title','Survived']].mean().plot(kind='bar',figsize=(10,7))
plt.xticks(rotation=0)


# __The 'Rarewoman' has 100% survival rate, that's amazing!!__

# __It's natural to assume that 'Pclass' also plays a big part, as we can see from the plot below. The females in class 3 have a survival rate of about 50%, while survival rateof females from class1/2 are much higher.__

# In[47]:


fig,axes=plt.subplots(2,3,figsize=(15,8))
Sex1=['male','female']
for i,ax in zip(Sex1,axes):
    for j,pp in zip(range(1,4),ax):
        PclassSex=combined_data[(combined_data.Sex==i)&(combined_data.Pclass==j)]['Survived'].value_counts().sort_index(ascending=False)
        pp.bar(range(len(PclassSex)),PclassSex,label=(i,'Class'+str(j)))
        pp.set_xticks((0,1))
        pp.set_xticklabels(('Survived','Dead'))
        pp.legend(bbox_to_anchor=(0.6,1.1))


# # Feature Engeneering

# In[48]:


# create age bands
combined_data.AgeCut=pd.cut(combined_data.Age,5)


# In[49]:


# create fare bands
combined_data.FareCut=pd.qcut(combined_data.Fare,5)


# In[50]:


combined_data.AgeCut.value_counts().sort_index()


# In[51]:


combined_data.FareCut.value_counts().sort_index()


# In[52]:


# replace agebands with ordinals
combined_data.loc[combined_data.Age<=16.136,'AgeCut']=1
combined_data.loc[(combined_data.Age>16.136)&(combined_data.Age<=32.102),'AgeCut']=2
combined_data.loc[(combined_data.Age>32.102)&(combined_data.Age<=48.068),'AgeCut']=3
combined_data.loc[(combined_data.Age>48.068)&(combined_data.Age<=64.034),'AgeCut']=4
combined_data.loc[combined_data.Age>64.034,'AgeCut']=5


# In[53]:


combined_data.loc[combined_data.Fare<=7.854,'FareCut']=1
combined_data.loc[(combined_data.Fare>7.854)&(combined_data.Fare<=10.5),'FareCut']=2
combined_data.loc[(combined_data.Fare>10.5)&(combined_data.Fare<=21.558),'FareCut']=3
combined_data.loc[(combined_data.Fare>21.558)&(combined_data.Fare<=41.579),'FareCut']=4
combined_data.loc[combined_data.Fare>41.579,'FareCut']=5


# __We can see from the plot that 'FareCut' has a big impact on survial rate.__

# In[54]:


combined_data[['FareCut','Survived']].groupby(['FareCut']).mean().plot.bar(figsize=(8,5))


# In[55]:


combined_data.corr()


# __We haven't gererate any feature from 'Parch','Pclass','SibSp','Title', so let's do this by using pivot table.__

# In[56]:


combined_data[combined_data.Survived.notnull()].pivot_table(index=['Title','Pclass'],values=['Survived']).sort_values('Survived',ascending=False)


# In[57]:


combined_data[combined_data.Survived.notnull()].pivot_table(index=['Title','Parch'],values=['Survived']).sort_values('Survived',ascending=False)


# #### _From the pivot tables above, there is definitely a relationship among 'Survived','Title','Pclass','Parch'. So we can combine them together._

# In[58]:


# only choose the object with not null 'Survived'.
TPP=combined_data[combined_data.Survived.notnull()].pivot_table(index=['Title','Pclass','Parch'],values=['Survived']).sort_values('Survived',ascending=False)
TPP


# In[59]:


TPP.plot(kind='bar',figsize=(16,10))
plt.xticks(rotation=40)
plt.axhline(0.8,color='#BA55D3')
plt.axhline(0.5,color='#BA55D3')
plt.annotate('80% survival rate',xy=(30,0.81),xytext=(32,0.85),arrowprops=dict(facecolor='#BA55D3',shrink=0.05))
plt.annotate('50% survival rate',xy=(32,0.51),xytext=(34,0.54),arrowprops=dict(facecolor='#BA55D3',shrink=0.05))


# __From the plot, we can draw some horizontal lines and make some classification. I only choose 80% and 50%, because I'm so afraid of overfitting.__

# In[60]:


# use 'Title','Pclass','Parch' to generate feature 'TPP'.
Tit=['Girl','Master','Mr','Miss','Mrs','Rareman','Rarewoman']
for i in Tit:
    for j in range(1,4):
        for g in range(0,10):
            if combined_data.loc[(combined_data.Title==i)&(combined_data.Pclass==j)&(combined_data.Parch==g)&(combined_data.Survived.notnull()),'Survived'].mean()>=0.8:
                combined_data.loc[(combined_data.Title==i)&(combined_data.Pclass==j)&(combined_data.Parch==g),'TPP']=1
            elif combined_data.loc[(combined_data.Title==i)&(combined_data.Pclass==j)&(combined_data.Parch==g)&(combined_data.Survived.notnull()),'Survived'].mean()>=0.5:
                combined_data.loc[(combined_data.Title==i)&(combined_data.Pclass==j)&(combined_data.Parch==g),'TPP']=2
            elif combined_data.loc[(combined_data.Title==i)&(combined_data.Pclass==j)&(combined_data.Parch==g)&(combined_data.Survived.notnull()),'Survived'].mean()>=0:
                combined_data.loc[(combined_data.Title==i)&(combined_data.Pclass==j)&(combined_data.Parch==g),'TPP']=3
            else: 
                combined_data.loc[(combined_data.Title==i)&(combined_data.Pclass==j)&(combined_data.Parch==g),'TPP']=4


# + __'TPP=1' means highest probability to survive, and 'TPP=3' means the lowest.__
# + __'TPP=4' means there is no such combination of (Title&Pclass&Pclass) in train set. Let's see what kind of combination it contains.__

# In[61]:


combined_data[combined_data.TPP==4]


# __ We can simply classify them by 'Sex'&'Pclass'.__

# In[62]:


combined_data.ix[(combined_data.TPP==4)&(combined_data.Sex=='female')&(combined_data.Pclass!=3),'TPP']=1
combined_data.ix[(combined_data.TPP==4)&(combined_data.Sex=='female')&(combined_data.Pclass==3),'TPP']=2
combined_data.ix[(combined_data.TPP==4)&(combined_data.Sex=='male')&(combined_data.Pclass!=3),'TPP']=2
combined_data.ix[(combined_data.TPP==4)&(combined_data.Sex=='male')&(combined_data.Pclass==3),'TPP']=3


# In[63]:


combined_data.TPP.value_counts()


# In[64]:


combined_data.info()


# # Basic Modeling & Evaluation

# In[65]:


predictors=['Cabin','Embarked','Parch','Pclass','Sex','SibSp','Title','AgeCut','TPP','FareCut','Age','Fare']


# In[66]:


# convert categorical variables to numerical variables
combined_data_dummies=pd.get_dummies(combined_data[predictors])


# In[67]:


combined_data_dummies.head()


# __We choose 7 models and use 5-folds cross-calidation to evaluate these models.__

# Models include:
# 
# + k-Nearest Neighbors
# + Logistic Regression
# + Naive Bayes classifier
# + Decision Tree
# + Random Forrest
# + Gradient Boosting Decision Tree
# + Support Vector Machine

# In[68]:


from sklearn.model_selection import cross_val_score


# In[69]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


# In[70]:


models=[KNeighborsClassifier(),LogisticRegression(),GaussianNB(),DecisionTreeClassifier(),RandomForestClassifier(),
       GradientBoostingClassifier(),SVC()]


# In[71]:


combined_data.shape,combined_data_dummies.shape


# In[72]:


X=combined_data_dummies[:891]
y=combined_data.Survived[:891]
test_X=combined_data_dummies[891:]


# __As some algorithms such as KNN and SVM are sensitive to the scaling of the data, here we also apply standard-scaling to the data.__

# In[73]:


from sklearn.preprocessing import StandardScaler


# In[74]:


scaler=StandardScaler()
X_scaled=scaler.fit(X).transform(X)
test_X_scaled=scaler.fit(X).transform(test_X)


# In[75]:


# evaluate models by using cross-validation
names=['KNN','LR','NB','Tree','RF','GDBT','SVM']
for name, model in zip(names,models):
    score=cross_val_score(model,X,y,cv=5)
    print("{}:{},{}".format(name,score.mean(),score))


# In[76]:


# used scaled data
names=['KNN','LR','NB','Tree','RF','GDBT','SVM']
for name, model in zip(names,models):
    score=cross_val_score(model,X_scaled,y,cv=5)
    print("{}:{},{}".format(name,score.mean(),score))


# __'k-Nearest Neighbors', 'Support Vector Machine' perform much better on scaled data__

# **Then we use (feature importances) in GradientBoostingClassifier to see which features are important.**

# In[77]:


model=GradientBoostingClassifier()


# In[78]:


model.fit(X,y)


# In[79]:


model.feature_importances_


# In[80]:


X.columns


# In[81]:


fi=pd.DataFrame({'importance':model.feature_importances_},index=X.columns)


# In[82]:


fi.sort_values('importance',ascending=False)


# In[83]:


fi.sort_values('importance',ascending=False).plot.bar(figsize=(11,7))
plt.xticks(rotation=30)
plt.title('Feature Importance',size='x-large')


# __Based on the bar plot, 'TPP','Fare','Age' are the most important.__

# **Now let's think through this problem in another way. Our goal here is to improve the overall accuracy. This is equivalent to minimizing the misclassified observations. So if all the misclassified observations are found, maybe we can see the pattern and generate some new features.**

# **Again we use cross-validation to search for the miscalssified observations**

# In[84]:


from sklearn.model_selection import KFold


# In[85]:


kf=KFold(n_splits=10,random_state=1)


# In[86]:


kf.get_n_splits(X)


# In[87]:


print(kf)


# In[88]:


# extract the indices of misclassified observations
rr=[]
for train_index, val_index in kf.split(X):
    pred=model.fit(X.ix[train_index],y[train_index]).predict(X.ix[val_index])
    rr.append(y[val_index][pred!=y[val_index]].index.values)


# In[89]:


rr


# In[90]:


# combine all the indices
whole_index=np.concatenate(rr)
len(whole_index)


# In[91]:


combined_data.ix[whole_index].head()


# In[92]:


diff=combined_data.ix[whole_index]


# In[93]:


diff.describe()


# In[94]:


diff.describe(include=['O'])


# In[95]:


# both mean and count of 'survived' should be considered.
diff.groupby(['Title'])['Survived'].agg([('average','mean'),('number','count')])


# In[96]:


diff.groupby(['Title','Pclass'])['Survived'].agg([('average','mean'),('number','count')])


# **It seems mainly the third class 'Miss'/'Mrs' and the first/third class 'Mr' are missclassified.**

# In[97]:


diff.groupby(['Title','Pclass','Parch','SibSp'])['Survived'].agg([('average','mean'),('number','count')])


# Gererally, we should only pick the categories with relatively large numbers. That is:

# 1. **'Mr','Pclass 1','Parch 0','SibSp 0', 17**
# 2. **'Mr','Pclass 1','Parch 0','SibSp 1', 8**
# 3. **'Mr','Pclass 2/3','Parch 0','SibSp 0', 32+7**
# 4. **'Miss','Pclass 3','Parch 0','SibSp 0', 21**

# __Then we add new feature 'MPPS'.__

# In[98]:


combined_data.loc[(combined_data.Title=='Mr')&(combined_data.Pclass==1)&(combined_data.Parch==0)&((combined_data.SibSp==0)|(combined_data.SibSp==1)),'MPPS']=1
combined_data.loc[(combined_data.Title=='Mr')&(combined_data.Pclass!=1)&(combined_data.Parch==0)&(combined_data.SibSp==0),'MPPS']=2
combined_data.loc[(combined_data.Title=='Miss')&(combined_data.Pclass==3)&(combined_data.Parch==0)&(combined_data.SibSp==0),'MPPS']=3
combined_data.MPPS.fillna(4,inplace=True)


# In[99]:


combined_data.MPPS.value_counts()


# From the __feature-Importance__ plot we can see the 'Fare' is the most important feature, let's explore whether we can generate some new feature.

# In[101]:


diff[(diff.Title=='Mr')|(diff.Title=='Miss')].groupby(['Title','Survived','Pclass'])[['Fare']].describe().unstack()


# In[102]:


combined_data[(combined_data.Title=='Mr')|(combined_data.Title=='Miss')].groupby(['Title','Survived','Pclass'])[['Fare']].describe().unstack()


# But there seems no big difference between the 'Fare' of 'diff' and 'combined_data'.

# __Finally we could draw a corrlelation heatmap__

# In[103]:


colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=20)
sns.heatmap(combined_data[['Cabin','Parch','Pclass','SibSp','AgeCut','TPP','FareCut','Age','Fare','MPPS','Survived']].astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# # Hyperparameters Tuning

# __Now let's do grid search for some algorithms. Since many algorithms performs better in scaled data, we will use scaled data.__

# In[104]:


predictors=['Cabin','Embarked','Parch','Pclass','Sex','SibSp','Title','AgeCut','TPP','FareCut','Age','Fare','MPPS']
combined_data_dummies=pd.get_dummies(combined_data[predictors])
X=combined_data_dummies[:891]
y=combined_data.Survived[:891]
test_X=combined_data_dummies[891:]

scaler=StandardScaler()
X_scaled=scaler.fit(X).transform(X)
test_X_scaled=scaler.fit(X).transform(test_X)


# In[105]:


from sklearn.model_selection import GridSearchCV


# ### k-Nearest Neighbors

# In[106]:


param_grid={'n_neighbors':[1,2,3,4,5,6,7,8,9]}
grid_search=GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)

grid_search.fit(X_scaled,y)

grid_search.best_params_,grid_search.best_score_


# ### Logistic Regression

# In[107]:


param_grid={'C':[0.01,0.1,1,10]}
grid_search=GridSearchCV(LogisticRegression(),param_grid,cv=5)

grid_search.fit(X_scaled,y)

grid_search.best_params_,grid_search.best_score_


# In[108]:


# second round grid search
param_grid={'C':[0.04,0.06,0.08,0.1,0.12,0.14]}
grid_search=GridSearchCV(LogisticRegression(),param_grid,cv=5)

grid_search.fit(X_scaled,y)

grid_search.best_params_,grid_search.best_score_


# ### Support Vector Machine

# In[109]:


param_grid={'C':[0.01,0.1,1,10],'gamma':[0.01,0.1,1,10]}
grid_search=GridSearchCV(SVC(),param_grid,cv=5)

grid_search.fit(X_scaled,y)

grid_search.best_params_,grid_search.best_score_


# In[110]:


#second round grid search
param_grid={'C':[2,4,6,8,10,12,14],'gamma':[0.008,0.01,0.012,0.015,0.02]}
grid_search=GridSearchCV(SVC(),param_grid,cv=5)

grid_search.fit(X_scaled,y)

grid_search.best_params_,grid_search.best_score_


# ### Gradient Boosting Decision Tree

# In[133]:


param_grid={'n_estimators':[30,50,80,120,200],'learning_rate':[0.05,0.1,0.5,1],'max_depth':[1,2,3,4,5]}
grid_search=GridSearchCV(GradientBoostingClassifier(),param_grid,cv=5)

grid_search.fit(X_scaled,y)

grid_search.best_params_,grid_search.best_score_


# In[134]:


#second round search
param_grid={'n_estimators':[100,120,140,160],'learning_rate':[0.05,0.08,0.1,0.12],'max_depth':[3,4]}
grid_search=GridSearchCV(GradientBoostingClassifier(),param_grid,cv=5)

grid_search.fit(X_scaled,y)

grid_search.best_params_,grid_search.best_score_


# # Ensemble Methods 

# ## Bagging

# __We use logistic regression with the parameter we just tuned to apply bagging.__

# In[135]:


from sklearn.ensemble import BaggingClassifier


# In[136]:


bagging=BaggingClassifier(LogisticRegression(C=0.06),n_estimators=100)


# ## VotingClassifier

# __We use five models to apply votingclassifier, namely logistic regression, random forest, gradient boosting decision,support vector machine and k-nearest neighbors.__

# In[115]:


from sklearn.ensemble import VotingClassifier


# In[137]:


clf1=LogisticRegression(C=0.06)
clf2=RandomForestClassifier(n_estimators=500)
clf3=GradientBoostingClassifier(n_estimators=120,learning_rate=0.12,max_depth=4)
clf4=SVC(C=4,gamma=0.015,probability=True)
clf5=KNeighborsClassifier(n_neighbors=8)


# In[117]:


eclf_hard=VotingClassifier(estimators=[('LR',clf1),('RF',clf2),('SVM',clf4),('KNN',clf5)])


# In[138]:


# add weights
eclfW_hard=VotingClassifier(estimators=[('LR',clf1),('RF',clf2),('SVM',clf4),('KNN',clf5)],weights=[1,1,2,2,1])


# In[139]:


# soft voting
eclf_soft=VotingClassifier(estimators=[('LR',clf1),('RF',clf2),('SVM',clf4),('KNN',clf5)],voting='soft')


# In[140]:


# add weights
eclfW_soft=VotingClassifier(estimators=[('LR',clf1),('RF',clf2),('SVM',clf4),('KNN',clf5)],voting='soft',weights=[1,1,2,2,1])


# __Finally we can evaluate all the models we just used.__

# In[141]:


models=[KNeighborsClassifier(n_neighbors=8),LogisticRegression(C=0.06),GaussianNB(),DecisionTreeClassifier(),RandomForestClassifier(n_estimators=500),
        SVC(C=4,gamma=0.015),
        eclf_hard,eclf_soft,eclfW_hard,eclfW_soft,bagging]


# In[142]:


names=['KNN','LR','NB','CART','RF','SVM','VC_hard','VC_soft','VCW_hard','VCW_soft','Bagging']
for name,model in zip(names,models):
    score=cross_val_score(model,X_scaled,y,cv=5)
    print("{}: {},{}".format(name,score.mean(),score))


# ## Stacking

# __We use logistic regression, k-nearest neighbors, support vector machine, Gradient Boosting Decision Tree as first-level models, and use random forest as second-level model.__

# In[143]:


from sklearn.model_selection import StratifiedKFold
n_train=train.shape[0]
n_test=test.shape[0]
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)  


# In[144]:


def get_oof(clf,X,y,test_X):
    oof_train=np.zeros((n_train,))
    oof_test_mean=np.zeros((n_test,))
    oof_test_single=np.empty((5,n_test))
    for i, (train_index,val_index) in enumerate(kf.split(X,y)):
        kf_X_train=X[train_index]
        kf_y_train=y[train_index]
        kf_X_val=X[val_index]
        
        clf.fit(kf_X_train,kf_y_train)
        
        oof_train[val_index]=clf.predict(kf_X_val)
        oof_test_single[i,:]=clf.predict(test_X)
    oof_test_mean=oof_test_single.mean(axis=0)
    return oof_train.reshape(-1,1), oof_test_mean.reshape(-1,1)


# In[145]:


LR_train,LR_test=get_oof(LogisticRegression(C=0.06),X_scaled,y,test_X_scaled)
KNN_train,KNN_test=get_oof(KNeighborsClassifier(n_neighbors=8),X_scaled,y,test_X_scaled)
SVM_train,SVM_test=get_oof(SVC(C=4,gamma=0.015),X_scaled,y,test_X_scaled)
GBDT_train,GBDT_test=get_oof(GradientBoostingClassifier(n_estimators=120,learning_rate=0.12,max_depth=4),X_scaled,y,test_X_scaled)


# In[146]:


X_stack=np.concatenate((LR_train,KNN_train,SVM_train,GBDT_train),axis=1)
y_stack=y
X_test_stack=np.concatenate((LR_test,KNN_test,SVM_test,GBDT_test),axis=1)


# In[147]:


X_stack.shape,y_stack.shape,X_test_stack.shape


# In[148]:


stack_score=cross_val_score(RandomForestClassifier(n_estimators=1000),X_stack,y_stack,cv=5)


# In[149]:


# cross-validation score of stacking
stack_score.mean(),stack_score


# In[150]:


pred=RandomForestClassifier(n_estimators=500).fit(X_stack,y_stack).predict(X_test_stack)


# In[151]:


tt=pd.DataFrame({'PassengerId':test.PassengerId,'Survived':pred})


# In[156]:


tt.to_csv('submission.csv',index=False)


# In[ ]:




