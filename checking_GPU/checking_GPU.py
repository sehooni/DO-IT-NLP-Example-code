# GPU가 설치된 컴퓨터 내에 CUDA가 정상적으로 작동하는가를 파악하는 코드
from pytorch_lightning.accelerators import gpu
import tensorflow as tf

print("GPU Available: ", tf.test.is_gpu_available())

from tensorflow.python.client import device_lib

device_lib.list_local_devices()

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

tf.debugging.set_log_device_placement(True)

a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)
