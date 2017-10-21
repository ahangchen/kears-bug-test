> 作者: [梦里茶](https://github.com/ahangchen)

### Keras中的Layer和Tensor
- Keras的最小操作单位是Layer，每次操作的是整个batch

定义如下一个InputLayer:

```python
a = Input(shape=(2,))
```

实际上相当于定义了一个shape为(batch_size, 2)大小的placeholder（如果不是InputLayer，就是定义了一个variable）

```python
a = tf.placeholder(tf.float32, shape=(batch_size, 2))
```

- Backend中Tensorflow的最小操作单位是Tensor，每次操作的数据是batch中的所有数据
定义如下一个Model，做简单的减法
```python
a = Input(shape=(2,))
b = Input(shape=(2,))
def minus(inputs):
    x, y = inputs
    return K.mean(x - y, axis=1) # 在batch里，对一个Input里的元素做平均
    # return K.mean(x - y) # 会在整个batch级别做平均
m = Lambda(minus, name='minus')([a, b])
model = Model(inputs=[a, b], outputs=[m])
```
    - 使用Lambda自定义Layer时，Lambda中使用Backend语法操作对象，操作的是Tensor
    - 当使用K.mean时，默认axis=None，会在整个batch级别做平均


完整的测试代码，可以尝试变更Lamda层的注释体会默认batch级别平均的坑爹之处
```python
from keras import Input
from keras import backend as K
from keras.engine import Model
from keras.layers import Lambda

import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

a = Input(shape=(2,))
b = Input(shape=(2,))

def minus(inputs):
    x, y = inputs
    return K.mean(x - y, axis=1)
    # return K.mean(x - y)

m = Lambda(minus, name='minus')([a, b])
model = Model(inputs=[a, b], outputs=[m])

v0 = np.array([1,2])
v1 = np.array([3,4])
v2 = np.array([5,6])
print model.predict([v0.reshape(1,2), v1.reshape(1,2)])
print model.predict([v0.reshape(1,2), v2.reshape(1,2)])
print model.predict([np.array([v0, v0]), np.array([v1, v2])])

```
