import time
import numpy as np
from numpy import cos, sin
from numpy.fft import fft, ifft, fft2, ifft2
from math import pi

def unring_1d(data, nsh, minW, maxW):
    # Unring on 1D data

    n = data.shape[1]
    numlines = data.shape[0]

    shifts = np.zeros([2 * nsh + 1], dtype=np.float64)
    shifts[0:nsh + 1] = np.arange(nsh + 1, dtype=np.float64)
    shifts[nsh + 1:] = -(np.arange(nsh, dtype=np.float64) + 1)

    phis = pi / n * shifts / nsh
    us = cos(phis) + 1j * sin(phis)

    sh = np.zeros([2*nsh+1,n],dtype=np.complex128)
    sh2 = np.zeros([2*nsh+1,n],dtype=np.complex128)

    maxn = (n - 1) / 2 if (n % 2 == 1) else n / 2 - 1
    data_out = np.empty(data.shape,dtype=np.complex128)
    line = np.empty(n,dtype=np.complex128)

    for k in range(numlines):
        line[:] = data[k, :]
        sh[0, :] = fft(line)
        sh[:, 0] = sh[0, 0]

        if n % 2 == 0:
            sh[:, n // 2] = 0
        es = np.zeros([2 * nsh + 1], dtype=np.complex128) +1
        for l in range(int(maxn)):
            es[1:] = es[1:] * us[1:]
            L = l + 1
            sh[1:, L] = es[1:] * sh[0, L]
            L = n - 1 - l
            sh[1:, L] = np.conjugate(es[1:]) * sh[0, L]

        for j in range(2 * nsh + 1):
            line2 = sh[j, :]
            sh2[j, :] = ifft(line2)

        TV1arr = np.zeros([2 * nsh + 1], dtype=np.double)
        TV2arr = np.zeros([2 * nsh + 1], dtype=np.double)

        for t in range(minW, maxW + 1):
            TV1arr[:] = TV1arr[:] + abs(sh2[:, (-t) % n].real - sh2[:, -(t + 1) % n].real)
            TV1arr[:] = TV1arr[:] + abs(sh2[:, (-t) % n].imag - sh2[:, -(t + 1) % n].imag)
            TV2arr[:] = TV2arr[:] + abs(sh2[:, t % n].real - sh2[:, (t + 1) % n].real)
            TV2arr[:] = TV2arr[:] + abs(sh2[:, t % n].imag - sh2[:, (t + 1) % n].imag)

        for l in range(n):

            minidx1 = np.argmin(TV1arr)
            minidx2 = np.argmin(TV2arr)

            if TV1arr[minidx1] < TV2arr[minidx2]:
                minidx = minidx1
            else:
                minidx = minidx2

            TV1arr[:] = TV1arr[:] + abs(sh2[:, (l - minW + 1) % n].real - sh2[:, (l - minW) % n].real)
            TV1arr[:] = TV1arr[:] - abs(sh2[:, (l - maxW) % n].real - sh2[:, (l - (maxW + 1)) % n].real)
            TV2arr[:] = TV2arr[:] + abs(sh2[:, (l + maxW + 1) % n].real - sh2[:, (l + maxW + 2) % n].real)
            TV2arr[:] = TV2arr[:] - abs(sh2[:, (l + minW) % n].real - sh2[:, (l + minW + 1) % n].real)

            TV1arr[:] = TV1arr[:] + abs(sh2[:, (l - minW + 1) % n].imag - sh2[:, (l - minW) % n].imag)
            TV1arr[:] = TV1arr[:] - abs(sh2[:, (l - maxW) % n].imag - sh2[:, (l - (maxW + 1)) % n].imag)
            TV2arr[:] = TV2arr[:] + abs(sh2[:, (l + maxW + 1) % n].imag - sh2[:, (l + maxW + 2) % n].imag)
            TV2arr[:] = TV2arr[:] - abs(sh2[:, (l + minW) % n].imag - sh2[:, (l + minW + 1) % n].imag)

            a0r = sh2[minidx, (l - 1) % n].real
            a1r = sh2[minidx, l].real
            a2r = sh2[minidx, (l + 1) % n].real
            a0i = sh2[minidx, (l - 1) % n].imag
            a1i = sh2[minidx, l].imag
            a2i = sh2[minidx, (l + 1) % n].imag

            s = np.double(shifts[minidx]) / nsh / 2.

            if s > 0:
                data_out[k, l] = (a1r * (1 - s) + a0r * s + 1j * (a1i * (1 - s) + a0i * s))
            else:
                s = -s
                data_out[k, l] = (a1r * (1 - s) + a2r * s + 1j * (a1i * (1 - s) + a2i * s))
    return data_out


def unring_2d(data1, nsh =25, minW = 1, maxW = 3):
    eps = 1E-10
    data1_a = np.empty((data1.shape[0], data1.shape[1]), dtype=np.complex128)
    data2_a = np.empty((data1.shape[1], data1.shape[0]), dtype=np.complex128)

    data1_a[:] = data1
    data2_a[:] = data1_a.transpose()

    tmp1 = fft2(data1_a)
    tmp2 = fft2(data2_a)

    cks = np.arange(data1.shape[0], dtype=np.float64)
    cks = (1 + cos(2 * pi * cks / data1.shape[0]))
    cjs = np.arange(data1.shape[1], dtype=np.float64)
    cjs = (1 + cos(2 * pi * cjs / data1.shape[1]))
    cks_plus_cjs = np.tile(cks, [data1.shape[1], 1]).transpose() + np.tile(cjs, [data1.shape[0], 1])
    cks_plus_cjs[cks_plus_cjs == 0] = eps

    tmp1 =  (tmp1 * np.tile(cks, [data1.shape[1], 1]).transpose())/ cks_plus_cjs
    tmp2 = (tmp2 * np.tile(cjs, [data1.shape[0], 1]).transpose())/ cks_plus_cjs

    data1_a[:] = ifft2(tmp1)
    data2_a[:] = ifft2(tmp2)

    data1b = unring_1d(data1_a, nsh, minW, maxW)
    data2b = unring_1d(data2_a, nsh, minW, maxW)

    tmp1[:] = fft2(data1b)
    tmp2[:] = fft2(data2b)

    tmp1[:] = (tmp1 + tmp2.transpose())

    tmp2 = ifft2(tmp1)
    tmp2 = tmp2.real

    tmp2[tmp2 < 0] = 0.
    return tmp2

# Uncomment if you want to run in one processor
# def unring_(arr, nsh=25, minW=1, maxW=3, out_dtype=None):
#     start_time = time.time()
#
#     if out_dtype is None:
#         out_dtype = arr.dtype
#
#     # We retain float64 precision, iff the input is in this precision:
#     if arr.dtype == np.float64:
#         calc_dtype = np.float64
#     # Otherwise, we'll calculate things in float32 (saving memory)
#     else:
#         calc_dtype = np.float32
#     arr = arr.astype(calc_dtype)
#
#     if arr.ndim > 4:
#         raise ValueError("Invalid Dimension",
#                          arr.shape)
#
#     unrang_arr = np.zeros(arr.shape, dtype=calc_dtype)
#
#     if arr.ndim == 3:
#         for k in range(arr.shape[2]):
#             slice_data = arr[:, :, k]
#             result_slice = unring_2d(slice_data, nsh, minW, maxW)
#             unrang_arr[:, :, k] = result_slice
#
#         print("--- %s seconds ---" % (time.time() - start_time))
#         return unrang_arr.astype(out_dtype)
#
#
#
#     for vol in range(arr.shape[3]):
#         for k in range(arr.shape[2]):
#             slice_data = arr[:, :, k, vol]
#             result_slice = unring_2d(slice_data, nsh, minW, maxW)
#             unrang_arr[:, :, k, vol] = result_slice
#
#     print("--- %s seconds ---" % (time.time() - start_time))
#     return unrang_arr.astype(out_dtype)





