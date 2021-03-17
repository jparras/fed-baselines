import datetime
import os
from source.utils import gpu_session
from source.constants import LOGGER_RESULT_FILE, ROOT_LOGGER_STR
import logging
import sys
from itertools import product
import tensorflow as tf
import json
from tensorboard.plugins.hparams import api as hp
from source.data_utils import batch_dataset
from source.experiment_utils import run_simulation, get_compiled_model_fn_from_dict
import gc


def create_hparams(hp_conf, data_set_conf, training_conf,
                   model_conf, logdir):
    exp_wide_keys = ["learning_rate", "l2_reg", "kl_weight", "batch_size",
                     "epochs_per_round", "hierarchical", "prior_scale", "natural_lr",
                     "server_learning_rate", "damping_factor"]
    HP_DICT = {}
    for key_0 in list(hp_conf.keys()) + exp_wide_keys:
        if (key_0 == 'learning_rate'
                or key_0 == 'kl_weight'
                or key_0 == 'l2_reg'
                or key_0 == 'server_learning_rate'
                or key_0 == 'natural_lr'
                or key_0 == 'damping_factor'
                or key_0 == 'delta_percentile'):
            HP_DICT[key_0] = hp.HParam(key_0, hp.RealInterval(0.0, 1e20))
        elif key_0 == 'batch_size':
            HP_DICT[key_0] = hp.HParam(key_0, hp.Discrete([1, 5, 10, 20, 40, 50,
                                                           64, 128, 256, 512]))
        elif key_0 == 'epochs_per_round':
            HP_DICT[key_0] = hp.HParam(key_0, hp.Discrete([1, 2, 5, 10, 15, 20,
                                                           25, 30, 35, 40,
                                                           45, 50, 55, 60,
                                                           65, 70, 75, 80,
                                                           85, 90, 95, 100,
                                                           110, 120, 130,
                                                           140, 150]))
        elif key_0 == 'clients_per_round':
            HP_DICT[key_0] = hp.HParam(key_0, hp.Discrete([1, 2, 3, 4, 5,
                                                           10, 15, 20, 50]))
        elif key_0 == 'method':
            HP_DICT[key_0] = hp.HParam(key_0, hp.Discrete(['virtual',
                                                           'fedprox']))
        elif key_0 == 'hierarchical':
            HP_DICT[key_0] = hp.HParam(key_0, hp.Discrete([True, False]))
        else:
            HP_DICT[key_0] = hp.HParam(key_0)
    for key, _ in data_set_conf.items():
        if key == 'name':
            HP_DICT[f'data_{key}'] = hp.HParam(f'data_{key}',
                                               hp.Discrete(['mnist',
                                                            'pmnist',
                                                            'femnist',
                                                            'shakespeare',
                                                            'human_activity',
                                                            'vehicle_sensor']))
        else:
            HP_DICT[f'data_{key}'] = hp.HParam(f'data_{key}')
    for key, _ in training_conf.items():
        HP_DICT[f'training_{key}'] = hp.HParam(f'training_{key}')
    for key, _ in model_conf.items():
        HP_DICT[f'model_{key}'] = hp.HParam(f'model_{key}')
    HP_DICT['run'] = hp.HParam('run')
    HP_DICT['config_name'] = hp.HParam('config_name')
    HP_DICT['training_num_rounds'] = hp.HParam('num_rounds',
                                               hp.RealInterval(0.0, 1e10))

    metrics = [hp.Metric('train/sparse_categorical_accuracy',
                         display_name='train_accuracy'),
               hp.Metric('train/max_sparse_categorical_accuracy',
                         display_name='train_max_accuracy'),
               hp.Metric('client_all/sparse_categorical_accuracy',
                         display_name='client_all_accuracy'),
               hp.Metric('client_all/max_sparse_categorical_accuracy',
                         display_name='max_client_all_accuracy'),
               hp.Metric('server/sparse_categorical_accuracy',
                         display_name='server_accuracy'),
               hp.Metric('server/max_sparse_categorical_accuracy',
                         display_name='max_server_accuracy'),
               hp.Metric('client_selected/sparse_categorical_accuracy',
                         display_name='client_selected_accuracy'),
               hp.Metric('client_selected/max_sparse_categorical_accuracy',
                         display_name='max_client_selected_max_accuracy')
               ]

    print('create_hp_fun ' + str(logdir))

    with tf.summary.create_file_writer(str(logdir)).as_default():
        hp.hparams_config(hparams=HP_DICT.values(),
                          metrics=metrics)
    return HP_DICT


def write_hparams(hp_dict, session_num, exp_conf, data_set_conf,
                  training_conf, model_conf, logdir_run, config_name):

    hparams = {'run': int(session_num), 'config_name': config_name}
    for key_0, value_0 in exp_conf.items():
        if isinstance(value_0, list):
            hparams[hp_dict[key_0]] = str(value_0)
        else:
            hparams[hp_dict[key_0]] = value_0
    for key_1, value_1 in data_set_conf.items():
        hparams[hp_dict[f'data_{key_1}']] = str(value_1)
    for key_2, value_2 in training_conf.items():
        if key_2 == 'num_rounds':
            hparams[hp_dict[f'training_{key_2}']] = value_2
        else:
            hparams[hp_dict[f'training_{key_2}']] = str(value_2)
    for key_3, value_3 in model_conf.items():
        if key_3 == 'layers':
            continue
        hparams[hp_dict[f'model_{key_3}']] = str(value_3)

    # Only concatenation of the name of the layers
    layers = ''
    for layer in model_conf['layers']:
        layers = layers + layer['name'] + '_'
    hparams[hp_dict['model_layers']] = layers[:-1]

    print('write_hp_fun ' + str(logdir_run))

    with tf.summary.create_file_writer(str(logdir_run)).as_default():
        hp.hparams(hparams)


def run_experiment(config, fede_train_data, fed_test_data, train_size, test_size):
    # Create GPU session
    gpu_session(config['session']['num_gpus'])

    # Run the experiment

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(current_time)
    print(config)
    # Configs
    data_set_conf = config['data_set_conf']
    training_conf = config['training_conf']
    model_conf = config['model_conf']
    hp_conf = config['hp']
    session_conf = config['session']
    if 'input_shape' in model_conf:
        model_conf['input_shape'] = tuple(model_conf['input_shape'])

    logdir = os.path.join(config["result_dir"],  'logs', config["config_name"] + str(current_time))
    os.makedirs(logdir, exist_ok=True)  # Create dir for logger
    logfile = os.path.join(logdir, LOGGER_RESULT_FILE)

    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler = logging.FileHandler(str(logfile))
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(f_format)

    root_logger = logging.getLogger(ROOT_LOGGER_STR)
    root_logger.handlers = []
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(f_handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)

    logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)

    logger.debug(f"Making grid of hyperparameters")
    keys, values = zip(*hp_conf.items())
    experiments = [dict(zip(keys, v)) for v in product(*values)]
    logger.debug(f"Grid done")
    for session_num, exp_conf in enumerate(experiments):
        all_params = {**data_set_conf,
                      **training_conf,
                      **model_conf,
                      **exp_conf,
                      **session_conf}
        training_conf['num_rounds'] = \
            int(all_params['tot_epochs_per_client'] * all_params['num_clients']
                / (all_params['clients_per_round']
                   * all_params['epochs_per_round']))
        all_params['num_rounds'] = training_conf['num_rounds']
        if all_params.pop('check_numerics', None):
            print('enabled check numerics')
            tf.debugging.enable_check_numerics()
        else:
            print('no debugging')

        if 'damping_factor' not in all_params and all_params['method'] == 'virtual':
            all_params['damping_factor'] = 1-all_params['server_learning_rate']
            hp_conf['damping_factor'] = all_params['damping_factor']
            exp_conf['damping_factor'] = all_params['damping_factor']

        print(all_params)
        # Log configurations
        logdir_run = os.path.join(logdir,  f'{session_num}_{current_time}')
        logger.info(f"saving results in {logdir_run}")
        HP_DICT = create_hparams(hp_conf, data_set_conf, training_conf,
                                 model_conf, logdir_run)

        write_hparams(HP_DICT, session_num, exp_conf, data_set_conf,
                      training_conf, model_conf, logdir_run, config[
                          'config_name'])

        #with open(logdir_run / 'config.json', 'w') as config_file:
        #    json.dump(config, config_file, indent=4)

        # Prepare dataset
        logger.debug(f'batching datasets')
        #seq_length = data_set_conf.get('seq_length', None)
        federated_train_data_batched = [batch_dataset(data, all_params['batch_size']) for data in fede_train_data]
        federated_test_data_batched = [batch_dataset(data, all_params['batch_size']) for data in fed_test_data]

        sample_batch = tf.nest.map_structure(lambda x: x.numpy(), iter(federated_train_data_batched[0]).next())

        # Run the experiment
        logger.info(f'Starting run {session_num} '
                    f'with parameters {all_params}...')
        model_fn = get_compiled_model_fn_from_dict(all_params, sample_batch)
        run_simulation(model_fn, federated_train_data_batched,
                       federated_test_data_batched, train_size, test_size,
                       all_params, logdir_run)
        tf.keras.backend.clear_session()
        gc.collect()

        logger.info("Finished experiment successfully")

