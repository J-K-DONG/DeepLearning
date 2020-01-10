import os
import matplotlib.pyplot as plt
from scipy.misc import imresize
import argparse

if __name__ == '__main__':
    image_num = 1234
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_num', type = int, help='number of input image', default=1000)
    arg = parser.parse_args()
    print("number of image is : " + str(arg.image_num))

    # root path depends on your computer
    root = "../data/face/img_align_celeba_202599_178_218/"
    dir_name = "face_align_64_64_" + str(arg.image_num) + "/"
    save_root = "../data/face_align/" + dir_name
    resize_size = 64

    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    if not os.path.isdir(save_root + dir_name):
        os.mkdir(save_root + dir_name)
    img_list = os.listdir(root)

    #print(img_list)
    # ten_percent = len(img_list) // 10
    #print("number of face images : " + str(len(arg.image_num)))


    for i in range(arg.image_num):
        img = plt.imread(root + str(img_list[i]))
        img = imresize(img, (resize_size, resize_size))
        plt.imsave(fname=save_root + dir_name + img_list[i], arr=img)
        # 1000 pictures
        if (i % (arg.image_num / 10)) == 0:
            print('%d images complete' % i)
