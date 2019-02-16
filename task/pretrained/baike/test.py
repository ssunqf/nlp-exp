#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-


import torch, numpy

x = torch.randn((1, 20, 1024))
y = torch.randn((1, 20, 1024))
z = torch.randn((1, 20, 1024))

if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    z = z.cuda()


def einsum_attention(query, key, value, num_head):
    with torch.no_grad():
        b, t, d = query.size()
        query = query.reshape(b, t, num_head, d//num_head)
        key = key.reshape(b, t, num_head, d//num_head)
        value = value.reshape(b, t, num_head, d//num_head)
        scores = torch.softmax(torch.einsum('bihd,bjhd->bhij', [query, key]), -1)
        return torch.einsum('bhij,bjhd->bihd', scores, value).reshape(b, t, d)


def naive_attention(query, key, value, num_head):
    with torch.no_grad():
        b, t, d = query.size()
        h, d = num_head, d//num_head
        hx = query.reshape(b, t, h, d).permute(0, 2, 1, 3).reshape(-1, t, d)
        hy = key.reshape(b, t, h, d).permute(0, 2, 3, 1).reshape(-1, d, t)
        hz = value.reshape(b, t, h, d).permute(0, 2, 1, 3).reshape(-1, t, d)
        scores = torch.softmax(torch.matmul(hx, hy), -1)

        return torch.matmul(scores, hz).reshape(b, h, t, d).permute(0, 2, 1, 3).reshape_as(value)


def performance(func1, func2, *args, **kwargs):
    time2 = timeit.default_timer()
    for _ in range(1000):
        _ = func1(*args, **kwargs)
    time3 = timeit.default_timer()

    time0 = timeit.default_timer()
    for _ in range(1000):
        _ = func2(*args, **kwargs)
    time1 = timeit.default_timer()

    return (func1(*args, **kwargs)-func2(*args, **kwargs)).abs().max().item(), time1-time0, time3-time2


print('diff: {}, einsum: {}, naive: {}'.format(*performance(einsum_attention, naive_attention, x, y, z, 8)))


def naive(I, C):
    # N^8 scaling
    return torch.einsum('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)


def optimized(I, C):
    # N^5 scaling
    K = torch.einsum('pi,ijkl->pjkl', C, I)
    K = torch.einsum('qj,pjkl->pqkl', C, K)
    K = torch.einsum('rk,pqkl->pqrl', C, K)
    K = torch.einsum('sl,pqrl->pqrs', C, K)
    return K


N = 10
C = torch.randn(N, N)
I = torch.randn(N, N, N, N)

print('diff: {}, naive: {}, optimized: {}'.format(*performance(naive, optimized, I, C)))