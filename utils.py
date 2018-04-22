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


def get_best_weights(path_weights, mode='acc', postfix='.h5'):
    if not os.path.isdir(path_weights):
        return None
    sub_files = os.listdir(path_weights)
    if not sub_files:
        return None
    target = sub_files[0]
    sub_files_with_metric = list(filter(lambda f: f.endswith(postfix) and f.__contains__('-'), sub_files))
    if sub_files_with_metric:
        try:
            weights_value = [file.replace(postfix, '').split('-')[-2:] for file in sub_files_with_metric]
            key_filename = 'filename'
            kw = ['loss', 'acc']
            weights_info = []
            for filename, value in zip(sub_files_with_metric, weights_value):
                item = dict((k, float(v)) for k, v in zip(kw, value))
                item[key_filename] = filename
                weights_info.append(item)
            if mode not in kw:
                mode = 'acc'
            if mode == 'loss':
                weights_info = list(sorted(weights_info, key=lambda x: x['loss']))
            elif mode == 'acc':
                weights_info = list(sorted(weights_info, key=lambda x: x['acc'], reverse=True))
            target = weights_info[0][key_filename]
            print('The best weights is %s, sorted by %s.' % (target, mode))
        except:
            print('Parse best weights failure, choose first file %s.' % target)
    else:
        print('No weights with metric found, choose first file %s.' % target)
    return os.path.join(path_weights, target)
