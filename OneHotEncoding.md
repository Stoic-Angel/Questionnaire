## One Hot Encoding

* Machine Learning algorithms cannot understand natural language. For enabling them to understand categorical data, we need to convert it into a numerical format. This is what one hot encoding does. It converts categories into binary vector representations that the algorithm can easily understand.

### Example:

* We have a dataframe of various college students, displaying their academic status out of three possible options - Pass, PassWithGrace, Fail.
* One hot encoding will convert this into a binary vector representation like this [1,0,0], [0,1,0], and [0,0,1]. This enables the algorithm to understand the data and make predictions based on it.
* One drawback that troubles us later is the curse of dimensionality because the number of features has increased dramatically.

```import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = {'student_id': [1, 2, 3, 4, 5],
          'status': ['Pass', 'PassWithGrace', 'Fail', 'Pass', 'Fail']}
df = pd.DataFrame(data)

encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(df[['status']])

categories = encoder.get_feature_names(['status'])
df_encoded = pd.DataFrame(encoded_data, columns=categories)

df_final = pd.concat([df, df_encoded], axis=1)```
