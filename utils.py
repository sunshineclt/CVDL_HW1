import os
import platform


def is_mac():
    return platform.system().lower().startswith('darwin')


def is_linux():
    return platform.system().lower().startswith('linux')


def context_creator(name, **kwargs):
    return {
        'weights': 'params/%s/{epoch:05d}-{val_loss:.4f}-{val_acc:.4f}.h5' % name,
        'summary': 'log/%s' % name,
        'predictor_cache_dir': 'cache/%s' % name,
        'load_imagenet_weights': is_linux(),
        'path_json_dump': 'eval_json/%s/result%s.json' % (
            name, ('_' + kwargs['policy']) if kwargs.__contains__('policy') else ''),
    }


def parse_weight_file(weights):
    if not weights or not weights.endswith('.h5') or not weights.__contains__('/') or not weights.__contains__('-'):
        return None
    try:
        weights_info = weights.split(os.path.sep)[-1].replace('.h5', '').split('-')
        if len(weights_info) != 3:
            return None
        epoch = int(weights_info[0])
        val_loss = float(weights_info[1])
        val_acc = float(weights_info[2])
        return epoch, val_loss, val_acc
    except Exception as e:
        raise Exception('Parse weights failure: %s', str(e))
