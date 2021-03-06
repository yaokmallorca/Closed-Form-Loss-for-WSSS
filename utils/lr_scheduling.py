def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=20000, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    for param_group in optimizer.param_groups:
        # param_group['lr'] = init_lr*(1 - float(iter)/max_iter)**power
        param_group['lr'] = init_lr
    return optimizer

def poly_lr_step_scheduler(optimizer, curr_lr, itr, steps, lr_decay_iter=1, max_iter=20000, power=0.9, gamma=0.1):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
        :param gamma is decreased rate
    """
    if itr % lr_decay_iter or itr > max_iter:
        return optimizer

    for param_group in optimizer.param_groups:
        if itr in steps:
            curr_lr = gamma*curr_lr
        param_group['lr'] = curr_lr
    return optimizer