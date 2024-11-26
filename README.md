# MINIPROJECT

******AI-Driven Acoustic Surveillance: Scream Detection for Crime Prevention******

**AIM:**

 To enhance public safety by detecting human screams in real-time. Utilizing Machine Learning (ML) and Deep Learning (DL) techniques, the system differentiates screams from other environmental sounds, providing rapid alerts to authorities to potentially prevent crimes or ensure quicker responses.

** Features**

Real-Time Detection: Continuous monitoring of public spaces for acoustic distress signals.
High Accuracy: Utilizes advanced ML and DL models for reliable scream classification.
Noise Robustness: Processes audio to minimize background noise and enhance detection in complex environments.
Immediate Alerts: Sends notifications to law enforcement and emergency services upon detection.

**Problem Statement**
 
Traditional crime detection systems often rely on visual surveillance and human intervention, overlooking critical audio cues like screams. This project addresses:

Limitations of Existing Systems: Reactive rather than proactive.
Screams as Indicators: Associating screams with danger to provide actionable insights.

ALGORITHM :

1.Data Collection
Datasets: Audio datasets including screams, background noise, and other environmental sounds.
Augmentation: Pitch shifting, time stretching, and noise addition to improve robustness.
2.Preprocessing
Noise reduction using audio filters.
Audio segmentation for frame-level analysis.
Extraction of features like Mel-Frequency Cepstral Coefficients (MFCCs).
3.ML Approach
Algorithms: SVM, Random Forest, KNN.

Features: Pitch, amplitude, frequency, and STFT.
4.DL Approach
CNN: For spectrogram analysis.
RNN/LSTM: Capturing sequential audio patterns.

PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
df = pd.read_csv('/content/Crimes_-_2001_to_Present.csv')
#shows number of rows and columns of dataset
df.shape
# info() shows properties of dataset - number of entries, list of columns and their data types, memory usage of dataframe
df.info()
# by default describe command shows result in scientific format. To change it set_format command can be used.
pd.set_option('display.float_format', '{:.2f}'.format)

# describe() shows calculated parameters for columns: count, mean, std, min, percentiles, max
df.describe()
# head(n) shows first n rows from dataset
df.head(5)
# tail(n) shows last n rows from dataset
df.tail(5)
# for deleting columns drop() command can be used
df = df.drop(['Location Description','Ward','Community Area','X Coordinate','Y Coordinate','Latitude','Longitude','Location'],axis=1)
# find number of NaN values for all columns
df.isna().sum()
# dropna() - delete rows with NaN values (without parameters will delete such rows for all columns)
df = df.dropna(subset=['District','Case Number'])
# check if there are duplicated entries
df.duplicated().sum()
# function for weekday names mapping
def weekday_mapping(weekday):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return days[weekday]

# I'd use pd.to_datetime here but it didn't work for some reason
df['Day'] = df['Date'].apply(lambda x: x[3:5]).astype(int)
df['Month'] = df['Date'].apply(lambda x: x[:2]).astype(int)
df['Weekday'] = pd.to_datetime(df[['Year', 'Month', 'Day']]).dt.dayofweek
df['Weekday_Text'] = df['Weekday'].apply(weekday_mapping)
df['Quarter'] = pd.to_datetime(df[['Year', 'Month', 'Day']]).dt.to_period('Q').dt.strftime('Q%q')
df['Date_new'] = df.apply(lambda row: f"{row['Year']:04d}-{row['Month']:02d}-{row['Day']:02d}", axis=1)
# counting total number of crimes of each type
df_by_type_total = df['Primary Type'].value_counts().sort_values()
# making a graph for all types
plt.pie(df_by_type_total.values, labels=df_by_type_total.index, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Distribution of crime types in total')
plt.show()
# total count for 5 top most frequent crimes
df_by_type_total.tail(5)
# making a graph of 5 most frequent crimes in total
plt.pie(df_by_type_total[-5:].values, labels=df_by_type_total[-5:].index, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Distribution of crime types in total for 5 top frequent types')
plt.show()
# creating pivot table of crime types by years
pivot_table_types_by_years = df.pivot_table(index='Year', columns='Primary Type', aggfunc='size', fill_value=0)
# calculate min, max, mean, median values for 5 most frequent types of crime
pivot_table_types_by_years.agg({'THEFT': ['min','max','mean','median'], 'BATTERY': ['min','max','mean','median'], 'CRIMINAL DAMAGE': ['min','max','mean','median'], 'NARCOTICS': ['min','max','mean','median'], 'ASSAULT': ['min','max','mean','median']})
# the same can be done via describe() function
pivot_table_types_by_years[['THEFT','BATTERY','CRIMINAL DAMAGE','NARCOTICS','ASSAULT']].describe()
# making graph of crimes per year
df_by_year = df['Year'].value_counts().sort_index(ascending=True)

plt.bar(df_by_year.index,df_by_year.values,label='Number of crimes')

plt.title("Number of crimes per year")
plt.xlabel("Years")
plt.ylabel("Number of crimes")

plt.legend()
plt.show()
# making of graph for top 5 crime types
plt.figure(figsize=(10, 6))
plt.plot(pivot_table_types_by_years.index, pivot_table_types_by_years[['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT']].values)

plt.xlabel('Years')
plt.ylabel('Count of crimes')
plt.title('Number of crimes by year for the five most common types of crimes')
plt.legend(pivot_table_types_by_years[['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT']])

plt.show()
# daily crime chart of theft
df_for_plot_theft_2022 = df[(df['Primary Type']=='THEFT') & (df['Year']==2022)]['Date_new'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.plot(df_for_plot_theft_2022.index, df_for_plot_theft_2022.values)

plt.xlabel('Days')
plt.ylabel('Count of crimes')
plt.title('The number of thefts crimes by day during 2022')

plt.show()
# daily crime chart of criminal damage
df_for_plot_crimdam_2022 = df[(df['Primary Type']=='CRIMINAL DAMAGE') & (df['Year']==2022)]['Date_new'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.plot(df_for_plot_crimdam_2022.index, df_for_plot_crimdam_2022.values)

plt.xlabel('Days')
plt.ylabel('Count of crimes')
plt.title('The number of criminal damages by day during 2022')

plt.show()
# daily crime chart of narcotics
df_for_plot_narcotics_2022 = df[(df['Primary Type']=='NARCOTICS') & (df['Year']==2022)]['Date_new'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.plot(df_for_plot_narcotics_2022.index, df_for_plot_narcotics_2022.values)

plt.xlabel('Days')
plt.ylabel('Count of crimes')
plt.title('The number of narcotics crimes by day during 2022')

plt.show()

# daily crime chart of assault
df_for_plot_assault_2022 = df[(df['Primary Type']=='ASSAULT') & (df['Year']==2022)]['Date_new'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.plot(df_for_plot_assault_2022.index, df_for_plot_assault_2022.values)

plt.xlabel('Days')
plt.ylabel('Count of crimes')
plt.title('The number of assault crimes by day during 2022')

plt.show()
# find 10 blocks with most number of crimes
df.groupby(['Block']).size().sort_values(ascending=False)[:10]
# making dataframe with required data
df_top_5_arrest = df[df['Primary Type'].isin(['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT'])].groupby(['Primary Type', 'Arrest']).size().reset_index(name='Count')
df_top_5_arrest['Arrest'] = df_top_5_arrest['Arrest'].astype(str)
# making pivot table
pivot_df_top_5_arrest = df_top_5_arrest.pivot(index='Primary Type', columns='Arrest', values='Count')# creating bar graph
plt.bar(pivot_df_top_5_arrest.index, pivot_df_top_5_arrest['False'], color='r')
plt.bar(pivot_df_top_5_arrest.index, pivot_df_top_5_arrest['True'], bottom=pivot_df_top_5_arrest['False'], color='g')

plt.xlabel('Primary_Type')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.ticklabel_format(style='plain', axis='y')
plt.title('Arrest distribution analysis')
plt.tight_layout()

plt.show()
# Basic Libraries

import pandas as pd
import numpy as np

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler
# Libraries for Classification and building Models
# Libraries for Classification and building Models

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC# Project Specific Libraries

import os
import librosa
import librosa.display
import glob
import skimagedf = pd.read_csv("/content/UrbanSound8K.csv")

'''We will extract classes from this metadata.'''

df.head()dat1, sampling_rate1 = librosa.load('/content/7061-6-0-0.wav')
dat2, sampling_rate2 = librosa.load('/content/7383-3-0-0.wav')plt.figure(figsize=(20, 10))
D = librosa.amplitude_to_db(np.abs(librosa.stft(dat1)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')plt.figure(figsize=(20, 10))
D = librosa.amplitude_to_db(np.abs(librosa.stft(dat2)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')'''EXAMPLE'''

dat1, sampling_rate1 = librosa.load('/content/9031-3-3-0.wav')
arr = librosa.feature.melspectrogram(y=dat1, sr=sampling_rate1)
arr.shapefeature = []
label = []

def parser(row):
    # Function to load files and extract features
    for i in range(8732):
        file_name = '../input/urbansound8k/fold' + str(df["fold"][i]) + '/' + df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        # We extract mfcc feature from data
        mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        feature.append(mels)
        label.append(df["classID"][i])
    return [feature, label]input_dim = (32, 32, 3)

model.add(Conv2D(64, (3, 3), padding = "same", activation = "tanh", input_shape = input_dim))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding = "same", activation = "tanh"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1024, activation = "tanh"))
model.add(Dense(10, activation = "softmax"))model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
predictions = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print(score)
preds = np.argmax(predictions, axis = 1)
result = pd.DataFrame(preds)
result.to_csv("UrbanSound8kResults.csv")

```
**OUTPUT:**

INPUT FREQUENCY IMAGES

![Screenshot 2024-11-26 103235](https://github.com/user-attachments/assets/0062737d-e19a-40e9-be0e-17afcf53b304)

![Screenshot 2024-11-26 103248](https://github.com/user-attachments/assets/ab74a236-9c6f-49b8-8d38-0081fd1d19d1)

PREDICTION ACCURACY :

![Screenshot 2024-11-26 103442](https://github.com/user-attachments/assets/4f9d51ef-65e5-4a04-8a3e-68a2d3c4af7b)


Results

Thus,the system  Achieves 85-95% accuracy in scream detection.
