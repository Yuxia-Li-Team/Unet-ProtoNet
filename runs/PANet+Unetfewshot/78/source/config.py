import os
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('PANet+Unet')
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def config():
    input_size = (1024, 1024)
    seed = 1234
    gpu_id = 0
    mode1 = 'FEWSHOT train'
    mode2 = 'CNN train'
    mode3 = 'test FEWSHOT'
    mode4 = 'test CNN'
    mode5 = 'FEWSHOT gen'
    mode = mode1

    if mode == 'FEWSHOT train':
        dataset = 'Mas_map'
        iters = 30000
        batch_size = 1
        lr_milestones = [10000, 20000, 30000]
        align_loss_scaling = 30
        print_interval = 50
        save_model_interval = 10000
        align = True
        task_setting = {
            'shots': 5,
            'queries': 3,
        }
    elif mode == 'CNN train':
        dataset = 'Mas_map'
        epoch = 2000
        batch_size = 16
        lr_milestones = [500, 1000, 1500, 2000]
        save_model_interval = [500, 1000, 1500, 2000]
        train_size = 5000
        test_size = 1226
        pretraining = False

    elif mode == 'test FEWSHOT':
        dataset = 'Mas_map'
        batch_size = 1
        load_path = ''
    elif mode == 'test CNN':
        dataset = 'Mas_map'
        batch_size = 16
        load_path = ''
    elif mode == 'FEWSHOT gen':
        pass
    else:
        raise ValueError('typed wrong string to set the mode')

    exp_name = ' '.join(dataset)

    path = {
        'log_dir': './runs',
        'init_path': './pretrained_model/vgg16-397923af.pth',
    }

@ex.config_hook
def observer(config, command_name, logger):
    exp_name = ex.path
    if config['mode'] == 'FEWSHOT train':
        exp_name += 'fewshot'
    elif config['mode'] == 'CNN train':
        exp_name += 'cnn'
    elif config['mode'] == 'test FEWSHOT':
        exp_name += 'fewshot test'
    elif config['mode'] == 'test CNN':
        exp_name += 'cnn test'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config