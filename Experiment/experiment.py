import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
from absl import app
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
memory_limit=4 * 1024

# 下面是我修改的自动判断cpu与gpu环境的代码，不用cpu时可以注释掉
if gpus:
    try:
        # ① 推荐：开启按需增长，避免一次性吃满显存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # ② 可选：如仍 OOM，再强制给 4096MB 上限（二选一，不要同时与①混用）
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
        )

        # 单卡不建议用 MirroredStrategy；若你原来用了，改为：
        # strategy = tf.distribute.OneDeviceStrategy("/GPU:0")

        # 双卡再开下面几行
        # tf.config.experimental.set_virtual_device_configuration(gpus[1], 
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
        print("[INFO] Using GPU:", gpus[0])
        # print("[INFO] Using GPU:", gpus[1])
    except Exception as e:
        print("[WARN] Failed to configure GPU, fallback to CPU:", e)
else:
    print("[INFO] No GPU detected; running on CPU.")

# 下面为原本代码中配置gpu的部分，一共配置了两张gpu
#tf.config.experimental.set_virtual_device_configuration(gpus[0], 
#    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])

# 若只有一张gpu，可以取消设置第二张gpu
# tf.config.experimental.set_virtual_device_configuration(gpus[1], 
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])

from . import run_maddpg
from . import run_mad4pg
from . import run_mad4pg_gnn
from . import run_mad4pg_dynenv
from . import run_mad4pg_gnn_dynenv
from . import run_optres_edge
from . import run_optres_local
from . import run_ra
from . import run_ddpg
from . import run_d4pg

if __name__ == '__main__':
    # app.run(run_ddpg.main)
    # app.run(run_d4pg.main)
    #app.run(run_maddpg.main)
    #app.run(run_mad4pg.main)
    #app.run(run_mad4pg_gnn.main)
    app.run(run_mad4pg_gnn_dynenv.main)
    #app.run(run_mad4pg_dynenv.main)
    # app.run(run_optres_local.main)
    # app.run(run_optres_edge.main)
    # app.run(run_ra.main)
    