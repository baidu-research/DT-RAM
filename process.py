import sys
import os

path = sys.argv[1]
labels = []
images = []
train_test_split = []

labels_path = os.path.join(path, 'image_class_labels.txt')
for label in open(labels_path):
    labels.append(label.strip().split()[1])
    
images_path = os.path.join(path, 'images.txt')
for image in open(images_path):
    images.append(image.strip().split()[1])
    
train_test_split_path = os.path.join(path, 'train_test_split.txt')
for ttsplit in open(train_test_split_path):
    train_test_split.append(ttsplit.strip().split()[1])
    
train_file = open('train.list', 'w')
test_file = open('val.list', 'w')
for i in range(len(labels)):
    image_path = os.path.join(path, 'images', images[i])
    if train_test_split[i] == '1':
        train_file.write(image_path + "\t" + labels[i] + "\n")
    else:
        test_file.write(image_path + "\t" + labels[i] + "\n")
train_file.close()
test_file.close()
