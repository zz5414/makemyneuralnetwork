import h5py
import imageio
import os
import matplotlib.pyplot as plt

with h5py.File('output.h5py', 'w') as hf:
    for file in os.listdir('temp_train_set'):
        img = imageio.imread(os.path.join('temp_train_set', file))
        hf.create_dataset(f'img/{file}', data=img, compression='gzip', compression_opts=9)



with h5py.File('output.h5py', 'r') as hf:
    for i in hf['img']:
        plt.imshow(hf['img'][i])
        plt.show()