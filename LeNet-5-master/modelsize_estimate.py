import torch
import torch.nn as nn
import numpy as np


def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    # print('Model {} : Number of params: {}'.format(model._get_name(), para))
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    # 把模型逐层取出
    mods = list(model.children())
    # for i, m in enumerate(model.children()):
    #     print(i)
    #     print(m)
    out_sizes = []

    c2_temp = None
    for i in range(len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        if i == 1:
            c2_temp = m(input_)
            out = m(input_)
        elif i == 2:
            out += c2_temp
        elif i == 4:
            input_ = input_.flatten(start_dim=1)
            out = m(input_)
        else:
            out = m(input_)
        # print(out.shape)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    # print('Model {} : Number of intermedite variables without backward: {}'.format(model._get_name(), total_nums))
    # print('Model {} : Number of intermedite variables with backward: {}'.format(model._get_name(), total_nums*2))
    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))

