'''
Written by Roozbeh Bazargani
Date 6/11/2019
'''

import cv2 as cv
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

dirname1 = 'SVD'
if os.path.exists(dirname1):
    shutil.rmtree(dirname1)
os.makedirs(dirname1)

dirname2 = 'DCT'
if os.path.exists(dirname2):
    shutil.rmtree(dirname2)
os.makedirs(dirname2)

cimg = cv.imread('lena_gray.png')
gray = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
imageSize = os.path.getsize('lena_gray.png')
# print(gray.shape)
# SVD ---------------------------
U, sigma, V = np.linalg.svd(gray)
svdSize = []
ks = list(range(5,131,25))
# print(U.shape, sigma.shape, V.shape)
for k in ks:
    img = np.matrix(U[:, :k]) * np.diag(sigma[:k]) * np.matrix(V[:k, :])
    cv.imwrite(os.path.join(dirname1, 'lena_gray'+str(k)+'.png'), img)
    svdSize.append(os.path.getsize(os.path.join(dirname1, 'lena_gray'+str(k)+'.png')))

plt.plot(ks, svdSize)

plt.xlabel(r'K', fontsize=16)
plt.ylabel(r'Image Size (Bytes)',fontsize=16)
plt.title(r"SVD  --  Size of real image is " + str(imageSize) + ' bytes',
fontsize=16, color='gray')
# Make room for the ridiculously large title.
plt.subplots_adjust(top=0.8)

plt.show()

# DCT ----------------------------
N = gray.shape[0] # image is square
T = np.matrix(np.zeros((N,N)))
for i in range(N):
    T[0,i] = 1 / np.sqrt(N)
for i in range(N):
    for j in range(1,N):
        T[j,i] = np.sqrt(2/N)*np.cos((2*j+1)*i*np.pi/(2*N))
# print(T) # NOT CORRECT!
M = gray # - 128
D = T * M * np.linalg.inv(T)
R = np.matrix(np.zeros((N,N)))

dctSize = []
ks = list(range(25,226,50))

for k in ks:
    for i in range(k):
        for j in range(k):
            R[i,j] = D[i,j]
    decom = np.matrix.round(np.linalg.inv(T) * R * T)  # + 128
    cv.imwrite(os.path.join(dirname2, 'lena_gray'+ str(k) + '.png'), decom)
    dctSize.append(os.path.getsize(os.path.join(dirname2, 'lena_gray'+ str(k) + '.png')))

plt.plot(ks, dctSize)

plt.xlabel(r'K', fontsize=16)
plt.ylabel(r'Image Size (Bytes)',fontsize=16)
plt.title(r"DCT  --  Size of real image is " + str(imageSize) + ' bytes',
fontsize=16, color='gray')
# Make room for the ridiculously large title.
plt.subplots_adjust(top=0.8)

plt.show()