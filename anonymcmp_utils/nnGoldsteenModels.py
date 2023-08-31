from tensorflow import keras

class GoldsteenBClassModel1(keras.Model):

  def __init__(self, *args, **kwargs):
    super(GoldsteenBClassModel1, self).__init__(*args, **kwargs)
    self.dense1 = keras.layers.Dense(124, activation='relu')
    self.dense2 = keras.layers.Dense(1, activation='sigmoid')

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)


class GoldsteenBClassModel2(keras.Model):

  def __init__(self, *args, **kwargs):
      super(GoldsteenBClassModel2, self).__init__(*args, **kwargs)
      self.dense1 = keras.layers.Dense(1024, activation='tanh')
      self.dense2 = keras.layers.Dense(512, activation='tanh')
      self.dense3 = keras.layers.Dense(256, activation='tanh')
      self.dense4 = keras.layers.Dense(1, activation='sigmoid')

  def call(self, inputs):
      x1 = self.dense1(inputs)
      x2 = self.dense2(x1)
      x3 = self.dense3(x2)
      return self.dense4(x3)


class Goldsteen4ClassModel(keras.Model):

  def __init__(self, *args, **kwargs):
    super(Goldsteen4ClassModel, self).__init__(*args, **kwargs)
    self.dense1 = keras.layers.Dense(124, activation='relu')
    self.dense2 = keras.layers.Dense(4, activation='softmax')

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)