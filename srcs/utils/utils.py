import hydra


def instantiate(config, *args, is_func=False, **kwargs):
    """
    wrapper function for hydra.utils.instantiate.
    1. return None if config.class is None
    2. return function handle if is_func is True
    """
    assert '_target_' in config, f'Config should have \'_target_\' for class instantiation.'
    target = config['_target_']
    if target is None:
        return None
    if is_func:
        # get function handle
        modulename, funcname = target.rsplit('.', 1)
        mod = import_module(modulename)
        func = getattr(mod, funcname)

        # make partial function with arguments given in config, code
        kwargs.update({k: v for k, v in config.items() if k != '_target_'})
        partial_func = partial(func, *args, **kwargs)

        # update original function's __name__ and __doc__ to partial function
        update_wrapper(partial_func, func)
        return partial_func
    return hydra.utils.instantiate(config, *args, **kwargs)