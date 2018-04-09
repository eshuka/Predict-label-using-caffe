import numpy as np
import caffe, os, cv2, detect

if __name__ == "__main__":

	MODEL_FILE = ' '#deploy.prototxt
	PRETRAINED = ' '#caffe_model
	imagenet_labels_filename = ' '#label.txt

	data_set = raw_input("folder : ")  #image folder
	files = os.listdir(data_set)
	files = np.sort(files)

	#input_gpu = int(raw_input(" gpu number : ")) #set gpu number
	caffe.set_mode_gpu()
	caffe.set_device(0)
	net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

	for cc, x in enumerate(files):
		file_id = x.split('.')[0]
		image_path = data_set + '/' + file_id
		images =os.listdir(image_path)
		for c, y in enumerate(images):
			image_id = y.split('.')[0]
			image_id_txt = image_path + '/' + image_id #+ '.txt'
			image_id = image_path + '/' + image_id +'.jpg'
			image_id_path = image_path +  '/'
			detect.detect(image_id, net, imagenet_labels_filename)