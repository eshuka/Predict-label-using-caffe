import caffe
from scipy.misc import imresize
import numpy as np


save_top = open("place_scene.txt",'w')

def detect(image_id, net, imagenet_labels_filename):

    img = caffe.io.load_image(image_id)
    img = imresize(img, [224, 224])
    img = img.astype(np.uint8)

    out = net.forward_all(data=np.asarray([img.transpose(2, 0, 1)]))

    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\s')
    top_k = net.blobs['prob'].data[0].flatten().argsort()[-1: -381 : -1]

    top_5(image_id, labels, out, top_k, save_top)  #extract top_5    
    #top_1(image_id, labels, out, top_k, save_top)  #extract top_1

def top_5(image_id, labels, out, top_k,save_top):
    for i in range(0, 5):    
        image_str = "{}".format(image_id)
        save_top.write(image_str)
        label_str = "{}".format(labels[top_k[i]])
        save_top.write(label_str)
    	probs = out['prob'][0][top_k[i]]
        probs_str = ", {}\n".format(probs)
    	save_top.write(probs_str)
    	print ""
    save_top.write('\n')

def top_1(image_id, labels, out, top_k,save_top):   
    image_str = "{}".format(image_id)
    save_top.write(image_str)
    label_str = "{}".format(labels[top_k[0]])
    save_top.write(label_str)
    probs = out['prob'][0][top_k[0]]
    probs_str = ", {}\n".format(probs)
    save_top.write(probs_str)
    print ""
    save_top.write('\n')
