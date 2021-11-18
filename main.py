import numpy as np
import pandas as pd
import tensorflow as tf

from model import NDAEModel as Model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

train = pd.read_csv("KDDTrain+.csv")
test = pd.read_csv("KDDTest+.csv")

# extracting numerical labels from categorical data

encoder = LabelEncoder()

train['protocol_type_label'] = encoder.fit_transform(train['protocol_type'])
test['protocol_type_label'] = encoder.fit_transform(test['protocol_type'])

train['service_label'] = encoder.fit_transform(train['service'])
test['service_label'] = encoder.fit_transform(test['service'])

train['flag_label'] = encoder.fit_transform(train['flag'])
test['flag_label'] = encoder.fit_transform(test['flag'])

train['attack_class'] = encoder.fit_transform(train['attack_class'])
test['attack_class'] = encoder.fit_transform(test['attack_class'])

# removing useless columns

train.drop(['num_learners'], axis = 1, inplace = True)
test.drop(['num_learners'], axis = 1, inplace = True)

# preparing data for training on models

#x_train = train.copy(deep = True)
train.drop(['protocol_type', 'service', 'flag'], axis = 1, inplace = True)

#x_test = test.copy(deep = True)
test.drop(['protocol_type', 'service', 'flag'], axis = 1, inplace = True)

train_target = train.pop('attack_class')
test_target = test.pop('attack_class')


model = Model().build_model()

train_dataset = tf.data.Dataset.from_tensor_slices((train.values, train_target.values))
test_dataset = tf.data.Dataset.from_tensor_slices((test.values, test_target.values))

train_dataset = train_dataset.shuffle(len(train)).batch(1024)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)

X_train = X_train.reshape(train.shape[0], 1, train.shape[1])
X_test = X_test.reshape(test.shape[0], 1, test.shape[1])

train_target = tf.keras.utils.to_categorical(train_target, num_classes=23)

print(X_train.shape)
print(X_test.shape)


history = model.fit(X_train, train_target, batch_size = 1024, epochs = 15, 
        verbose = 1)
model.save('NDAE')