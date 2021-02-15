from tmrl import TrainingOffline
import tmrl.custom.config_constants as cfg
from tmrl.envs import UntouchedGymEnv
from tmrl.sac import Agent as SAC_Agent
from tmrl.drtac import Agent as DCAC_Agent
from tmrl.custom.custom_dcac_interfaces import Tm20rtgymDcacInterface
from tmrl.util import partial
from tmrl.sac_models import Mlp, MlpPolicy
from tmrl.drtac_models import Mlp as SV_Mlp
from tmrl.drtac_models import MlpPolicy as SV_MlpPolicy
from tmrl.custom.custom_models import Tm_hybrid_1, TMPolicy
from tmrl.custom.custom_gym_interfaces import TM2020InterfaceLidar, TMInterfaceLidar, TM2020Interface, TMInterface, CogniflyInterfaceTask1
from tmrl.custom.custom_memories import get_local_buffer_sample, MemoryTMNFLidar, MemoryTMNF, MemoryTM2020, get_local_buffer_sample_tm20_imgs, get_local_buffer_sample_cognifly, MemoryCognifly, TrajMemoryTMNFLidar
from tmrl.custom.custom_preprocessors import obs_preprocessor_tm_act_in_obs, obs_preprocessor_tm_lidar_act_in_obs, obs_preprocessor_cognifly
from tmrl.custom.custom_checkpoints import load_run_instance_images_dataset, dump_run_instance_images_dataset
import numpy as np


# MODEL, GYM ENVIRONMENT, REPLAY MEMORY AND TRAINING: ===========

if cfg.PRAGMA_DCAC:
    TRAIN_MODEL = SV_Mlp
    POLICY = SV_MlpPolicy
else:
    TRAIN_MODEL = Mlp if cfg.PRAGMA_LIDAR else Tm_hybrid_1
    POLICY = MlpPolicy if cfg.PRAGMA_LIDAR else TMPolicy

if cfg.PRAGMA_LIDAR:
    INT = partial(TM2020InterfaceLidar, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD) if cfg.PRAGMA_TM2020_TMNF else partial(TMInterfaceLidar, img_hist_len=cfg.IMG_HIST_LEN)
else:
    INT = partial(TM2020Interface, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD) if cfg.PRAGMA_TM2020_TMNF else partial(TMInterface, img_hist_len=cfg.IMG_HIST_LEN)
CONFIG_DICT = {
    "interface": INT,
    "time_step_duration": 0.05,
    "start_obs_capture": 0.04,  # /!\ lidar capture takes 0.03s
    "time_step_timeout_factor": 1.0,
    "ep_max_length": np.inf,
    "real_time": True,
    "async_threading": True,
    "act_in_obs": True,  # ACT_IN_OBS
    "act_buf_len": cfg.ACT_BUF_LEN,
    "benchmark": cfg.BENCHMARK,
    "wait_on_done": True,
}

# to compress a sample before sending it over the local network/Internet:
SAMPLE_COMPRESSOR = get_local_buffer_sample if cfg.PRAGMA_LIDAR else get_local_buffer_sample_tm20_imgs
# to preprocess observations that come out of the gym environment and of the replay buffer:
OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_act_in_obs if cfg.PRAGMA_LIDAR else obs_preprocessor_tm_act_in_obs
# to augment data that comes out of the replay buffer (applied after observation preprocessing):
SAMPLE_PREPROCESSOR = None

if cfg.PRAGMA_LIDAR:
    MEM = TrajMemoryTMNFLidar if cfg.PRAGMA_DCAC else MemoryTMNFLidar
else:
    assert not cfg.PRAGMA_DCAC, "DCAC not implemented here"
    MEM = MemoryTM2020 if cfg.PRAGMA_TM2020_TMNF else MemoryTMNF

MEMORY = partial(MEM,
                 path_loc=cfg.DATASET_PATH,
                 imgs_obs=cfg.IMG_HIST_LEN,
                 act_buf_len=cfg.ACT_BUF_LEN,
                 obs_preprocessor=OBS_PREPROCESSOR,
                 sample_preprocessor=None if cfg.PRAGMA_DCAC else SAMPLE_PREPROCESSOR,
                 crc_debug=cfg.CRC_DEBUG
                 )

# ALGORITHM: ===================================================

if cfg.PRAGMA_DCAC:  # DCAC
    AGENT = partial(DCAC_Agent,
                    Interface=Tm20rtgymDcacInterface,
                    OutputNorm=partial(beta=0., zero_debias=False),
                    device='cuda' if cfg.PRAGMA_CUDA_TRAINING else 'cpu',
                    Model=partial(TRAIN_MODEL, act_buf_len=cfg.ACT_BUF_LEN),
                    lr=0.0003,  # default 0.0003
                    discount=0.995,  # default and best tmnf so far: 0.99
                    target_update=0.005,
                    reward_scale=2.0,  # 2.0,  # default: 5.0, best tmnf so far: 0.1, best tm20 so far: 2.0
                    entropy_scale=1.0)  # default: 1.0),  # default: 1.0
else:  # SAC
    AGENT = partial(SAC_Agent,
                    OutputNorm=partial(beta=0., zero_debias=False),
                    device='cuda' if cfg.PRAGMA_CUDA_TRAINING else 'cpu',
                    Model=partial(TRAIN_MODEL, act_buf_len=cfg.ACT_BUF_LEN),
                    lr=0.0003,  # default 0.0003
                    discount=0.99,  # default and best tmnf so far: 0.99
                    target_update=0.005,
                    reward_scale=2.0,  # 2.0,  # default: 5.0, best tmnf so far: 0.1, best tm20 so far: 2.0
                    entropy_scale=1.0)  # default: 1.0),  # default: 1.0

# TRAINER: =====================================================

if cfg.PRAGMA_LIDAR:  # lidar
    TRAINER = partial(TrainingOffline,
                      Env=partial(UntouchedGymEnv, id="rtgym:real-time-gym-v0", gym_kwargs={"config": CONFIG_DICT}),
                      Memory=MEMORY,
                      memory_size=1000000,
                      batchsize=128,
                      epochs=400,  # 400
                      rounds=10,  # 10
                      steps=1000,  # 1000
                      update_model_interval=1000,
                      update_buffer_interval=1000,
                      max_training_steps_per_env_step=1.0,
                      profiling=cfg.PROFILE_TRAINER,
                      Agent=AGENT)
else:  # images
    TRAINER = partial(TrainingOffline,
                      Env=partial(UntouchedGymEnv, id="rtgym:real-time-gym-v0", gym_kwargs={"config": CONFIG_DICT}),
                      Memory=MEMORY,
                      memory_size=1000000,
                      batchsize=128,  # 128
                      epochs=100000,  # 10
                      rounds=10,  # 50
                      steps=50,  # 2000
                      update_model_interval=50,
                      update_buffer_interval=1,
                      max_training_steps_per_env_step=1.0,
                      profiling=cfg.PROFILE_TRAINER,
                      Agent=AGENT)

# CHECKPOINTS: ===================================================

DUMP_RUN_INSTANCE_FN = None if cfg.PRAGMA_LIDAR else dump_run_instance_images_dataset
LOAD_RUN_INSTANCE_FN = None if cfg.PRAGMA_LIDAR else load_run_instance_images_dataset