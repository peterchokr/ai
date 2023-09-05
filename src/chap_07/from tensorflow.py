from tensorflow.keras import layers
from tensorflow.keras import initializers

layer = layers.Dense(
    units = 64,
    kernel_initializer = initializers.RandomNormal(stddev=0.01),
    bias_initializer = initializers.Zeros()
)


layer = layers.Dense(
    units = 64,
    kernel_initializer = 'random_normal',
    bias_initializer = 'zeros'
)