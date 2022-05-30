from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import json
import numpy as np
import os
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import copy
import torchvision
import cv2 as io
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.distance import directed_hausdorff
import torchvision.transforms.functional as F
import math
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
import csv

def Append2Frame():

    list = []
    with open(".\sose2021_iiml_mura-seminar\LatFile_metal_equally.json",'r') as f:
        png_list = json.load(f)
    #print(len(png_list))   390
    for i in range(len(png_list)):
        try:
            with open('.\\sose2021_iiml_mura-seminar\\MURA-v1.1\\train\\{}'.format(png_list[i]).replace('png', 'json'), 'r') as f:
                json_info = json.load(f)
        except:
            continue
                 
        for entry in json_info['shapes']:
            if entry['label'] == 'Radius Shaft Center Line':
                RSCL_point_list = entry['points']
                RSCL_point_list = sorted(RSCL_point_list, key=(lambda x:x[1]))
                n = len(RSCL_point_list)
                if n > 2:#to ensure only two points available
                    num_delete = n - 2
                    for j in range(num_delete):   
                        del RSCL_point_list[1]
                assert len(RSCL_point_list) == 2
            
            
            if entry['label'] == 'Wrist Joint Line':
                WJL_point_list = entry['points']
                WJL_point_list = sorted(WJL_point_list, key=(lambda x:x[0]))
                n = len(WJL_point_list)
                if n > 2:
                    num_delete = n - 2
                    for j in range(num_delete):   
                        del WJL_point_list[1]
                assert len(WJL_point_list) == 2

                
        list.append([id, '.\\Origin\\train\\{}'.format(png_list[i]),
                        RSCL_point_list[0][0], RSCL_point_list[0][1],
                        RSCL_point_list[1][0], RSCL_point_list[1][1],
                        WJL_point_list[0][0], WJL_point_list[0][1],
                        WJL_point_list[1][0], WJL_point_list[1][1]]
                        )
    print("list length: ", len(list))
    frame = None
    #load coordinates of four points
    frame = pd.DataFrame(list, columns = ['id', 'image_path', 'RSCL_0 x', 'RSCL_0 y', 'RSCL_1 x', 'RSCL_1 y',
                                                              'WJL_0 x', 'WJL_0 y', 'WJL_1 x', 'WJL_1 y'])
    return frame


class RadiusLandmarksDataset(Dataset):

    def __init__(self, data_frame, transform=None, output_shape=(4, 2)):

        self.landmarks_frame = data_frame
        self.transform = transform
        self.output_shape = output_shape

    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.landmarks_frame.iloc[idx]['image_path'])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 2:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(self.output_shape)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToPILImage(object):
  def __init__(self):
    self.transform = transforms.ToPILImage()
  def __call__(self, sample):
    image, landmarks = sample['image'], sample['landmarks']
              
    return {'image': self.transform(image), 'landmarks': landmarks}

class Normalize(object):
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        img_max = torch.max(image)
        img_min = torch.min(image)

        img = (image - img_min)/(img_max - img_min)
        
        return {'image': img, 'landmarks': landmarks}


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        w, h = image.size
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        trnsf = transforms.Resize((self.output_size[0], self.output_size[1]))
        img = trnsf(image)

        landmarks = landmarks * [new_w / w, new_h / h]
        # for i in range(4):
        #     landmarks[i][0] *= ratio_w
        #     landmarks[i][1] *= ratio_h

        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):

    def __init__(self, max_crop_factor = 0.9, prob = 0.5):
        self.prob = prob
        self.max_crop = max_crop_factor

    def __call__(self, sample):
        p = torch.rand(1)
        if p > self.prob:
          return sample 

        image, landmarks = sample['image'], sample['landmarks']

        w, h = image.size
        h, w = int(h), int(w)
        v = (self.max_crop + torch.randn(1) % (1-self.max_crop))
        new_h, new_w = h * v, w * v
        new_h, new_w = int(new_h.item()), int(new_w.item())
        top = int(torch.randint(0, h - new_h, [1]))
        left = int(torch.randint(0, w - new_w, [1]))

        image = image.crop((left, top, left + new_w, top + new_h))
        
        # for i in range(4):
        #     landmarks[i][0] -= left
        #     landmarks[i][1] -= top
        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class ColorJitter(object):

  def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
    self.transform = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

  def __call__(self, sample):
    image, landmarks = sample['image'], sample['landmarks']

    image = self.transform(image)
    return {'image': image,  ## hacky! Do this better in your code by arranging the augmentation in the right order!
            'landmarks': landmarks}

class Contrast_Enhance(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        gaussian_3 = io.GaussianBlur(image, (0, 0), 2.0)
        unsharp_image = io.addWeighted(image, self.alpha, gaussian_3, self.beta, 0, image)
        return {'image': unsharp_image, 'landmarks': landmarks}

class Contrast(object):
    def __init__(self):
        self.alpha = 1.5
        self.beta = 0.5

    def __call__(self, sample):
        seed = int(torch.rand(1) * 100)
        rng = np.random.default_rng(seed)
        self.alpha = 2.7*rng.random() + 0.3
        self.beta = 0.5*rng.random()
        image, landmarks = sample['image'], sample['landmarks']
        adjusted = io.convertScaleAbs(image, alpha=self.alpha, beta=self.beta)

        return {'image': adjusted, 'landmarks': landmarks}
    
class Random_Rotate(object):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black.
    """
    def __init__(self):
        self.angle = 0.0

    def __call__(self, sample):
        seed = int(torch.rand(1) * 100)
        rng = np.random.default_rng(seed)
        self.angle = rng.integers(low=-60, high=60, size=1)
        self.angle = float(self.angle)
        image, landmarks = sample['image'], sample['landmarks']
        # grab the dimensions of the image and then determine the
        # centre
        h, w = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = io.getRotationMatrix2D((cX, cY), self.angle, 1.0)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        image = io.warpAffine(image, M, (nW, nH))

        M2 = np.eye(3)
        M2[:2, :] = M

        a = np.ones(4).reshape(4, 1)
        tmp1 = np.hstack((landmarks, a))
        tmp = np.dot(M2, tmp1.transpose())
        landmarks = (tmp[:2, :] / tmp[-1, :]).transpose()

        #    image = cv2.resize(image, (w,h))
        return {'image': image, 'landmarks': landmarks}

class Addnoisy(object):
    def __init__(self, noisy):
        self.noisyType = noisy

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        row, col = image.shape[:2]
        seed = int(torch.rand(1) * 100)
        rng = np.random.default_rng(seed)
        mean = 0
        var = rng.random()
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)

        noisy = np.zeros(image.shape, np.float32)
        if len(image.shape) == 2:
            noisy = image + gauss
        else:
            noisy[:, :, 0] = image[:, :, 0] + gauss
            noisy[:, :, 1] = image[:, :, 1] + gauss
            noisy[:, :, 2] = image[:, :, 2] + gauss

        io.normalize(noisy, noisy, 0, 255, io.NORM_MINMAX, dtype=-1)
        noisy_image = noisy.astype(np.uint8)
        return {'image': noisy_image, 'landmarks': landmarks}

class Edge_Enhance(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        gaussian_3 = io.GaussianBlur(image, (0, 0), 2.0)
        unsharp_image = io.addWeighted(image, self.alpha, gaussian_3, self.beta, 0, image)
        return {'image': unsharp_image, 'landmarks': landmarks}

class Random_Shift(object):
    def __init__(self):
        self.shift = [0, 0]

    def __call__(self, sample):
        seed = int(torch.rand(1) * 100)
        rng = np.random.default_rng(seed)
        self.shift[0] = rng.integers(low=-100, high=100, size=1)
        self.shift[1] = rng.integers(low=-100, high=100, size=1)
        img, landmarks = sample['image'], sample['landmarks']
        # Translation matrix
        M = np.float32([[1, 0, self.shift[0]], [0, 1, self.shift[1]]])

        try:
            rows, cols = img.shape[:2]

            # warpAffine does appropriate shifting given the
            # translation matrix.
            res = io.warpAffine(img, M, (cols, rows))
            M2 = np.eye(3).astype('float')
            M2[:2, :] = M
            a = np.ones(4).reshape(4, 1).astype('float')
            tmp1 = np.hstack((landmarks, a))
            tmp = np.dot(M2, tmp1.transpose())
            landmarks = (tmp[:2, :] / tmp[-1, :]).transpose()

            return {'image': res, 'landmarks': landmarks}

        except IOError:
            print('Error while reading files !!!')

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, sample):
        p = torch.rand(1)
        if p > self.prob:
           return sample
        else:
            image, landmarks = sample['image'], sample['landmarks']
            image = F.hflip(img=image)
            w, h = image.size
            new_landmarks = landmarks + [-w, 0]
            new_landmarks = np.absolute(new_landmarks)

            return {'image': image, 'landmarks': new_landmarks}

class ToTensor(object):
    
    def __init__(self):
      self.transform = transforms.ToTensor()
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1)).astype('float64')
        # for i in range(4):
        #     landmarks[i][0] /= np.array(image.size[0], dtype=float)
        #     landmarks[i][1] /= np.array(image.size[1], dtype=float)
        landmarks /= np.array([image.size[1],image.size[0]], dtype=float)
        landmarks = landmarks.reshape(8)
        return {'image': self.transform(image),
                'landmarks': torch.from_numpy(landmarks)}

last_activation = torch.nn.Identity()

#Qi
def compute_euclidean_distance(labels_in, preds_in, input):
    w, h = input.shape[2:]
    max_d = []
    min_d = []
    mean_d = []
    distances = []
    labels1 = labels_in.cpu().detach().numpy()
    preds1 = preds_in.cpu().detach().numpy()

    for i in range(labels1.shape[0]):
        label = labels1[i].reshape(-1, 2)
        pred = preds1[i].reshape(-1, 2)
        label[:, 0] = label[:, 0] * w
        label[:, 1] = label[:, 1] * h
        pred[:, 0] = pred[:, 0] * w
        pred[:, 1] = pred[:, 1] * h
        # dis_1 = directed_hausdorff(label, pred)[0]
        for c in range(label.shape[0]):
            dis_1 = distance.euclidean(u=label[c], v=pred[c])
            dis_2 = np.linalg.norm(label[c] - pred[c])
            dis_3 = directed_hausdorff(pred, label)[0]
            distances.append(dis_1)

        mean_d.append(np.mean(distances))
        max_d.append(np.max(distances))
        min_d.append(np.min(distances))
        distances.clear()

    mean_distance = np.mean(mean_d)
    max_distance = np.mean(max_d)
    min_distance = np.mean(min_d)
    return mean_distance, max_distance, min_distance

def compute_boehler_angle(input_data, image):

    outputs = input_data
    width = image.shape[2]
    height = image.shape[3]
    b_angle_list = []

    for i in range(input_data.shape[0]):
        point_list = input_data.tolist()[i]
        point_x = point_list[::2]
        point_x_new = [i * width for i in point_x]
        point_x_new = list(map(int, point_x_new))  # transfer to int
        point_y = point_list[1::2]
        point_y_new = [i * height for i in point_y]
        point_y_new = list(map(int, point_y_new))

        # compute angle
        AB = [point_x_new[0], point_y_new[0], point_x_new[1], point_y_new[1]]
        CD = [point_x_new[2], point_y_new[2], point_x_new[3], point_y_new[3]]

        # vector 1 and vector 2 with form [point1_x, point1_y, point2_x, point2_y]
        dx1 = AB[2] - AB[0]
        dy1 = AB[3] - AB[1]
        dx2 = CD[2] - CD[0]
        dy2 = CD[3] - CD[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = float(angle1 * 180 / math.pi)
        # print(angle1)
        angle2 = math.atan2(dy2, dx2)
        angle2 = float(angle2 * 180 / math.pi)
        # print(angle2)
        if angle1 * angle2 >= 0:
            included_angle = abs(angle1 - angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        # included_angle is the angle between the vectors AB and CD as WJL and RCSL
        b_angle = abs(90 - included_angle)
        b_angle_list.append(b_angle)

    return b_angle_list

def compute_angle_error(angles_pred, angles_label):

    preds = np.array(angles_pred)
    labels = np.array(angles_label)

    # mean squared angle error per sample
    error_list = (preds - labels)**2
    mse1 = np.sum(error_list) / len(labels)
    mse2 = mean_squared_error(y_true=labels, y_pred=preds)

    return mse1, mse2, error_list

def compute_mean_distance(labels, preds, input):
  w, h = input.shape[2:]
  labels1 = labels.cpu()
  labels1 = labels1.detach().numpy()
  preds1 = preds.cpu()
  preds1 = preds1.detach().numpy()
  dis_matrix = np.zeros((labels1.shape[0], 4))
  for i in range(labels1.shape[0]):
    label = labels1[i].reshape(-1, 2)
    pred = preds1[i].reshape(-1, 2)
    label[:, 0] = label[:, 0] * w
    label[:, 1] = label[:, 1] * h
    pred[:, 0] = pred[:, 0] * w
    pred[:, 1] = pred[:, 1] * w
    distance = np.sqrt(np.sum(np.square(label - pred), axis=1))
    dis_matrix[i, :] = distance

  mean_dis = np.mean(dis_matrix, axis=0)
  mean_dis2 = np.mean(mean_dis)

  return mean_dis2, mean_dis


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25):
    writer = SummaryWriter()
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000
    best_distance = 10000
    best_dis_1 = 0.0

    print()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_mean_dis = 0.0
            running_max_dis = 0.0
            running_min_dis = 0.0
            running_corrects = 0
            angle_error_list = []
            angles_pred_list = []
            angles_label_list = []
            epoch_iter = 0

            running_distance = 0.0
            running_dis1 = 0.0

            # Iterate over data.
            for sample in dataloaders[phase]:
                inputs = sample['image'].float().to(device)
                labels = sample['landmarks'].float().to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = last_activation(outputs)
                    loss = criterion(preds, labels)

                    # metrics calculation
                    # min_dis, dis1, dis2 = compute_hausdorff_distance(labels, preds, inputs)
                    mean_dis, max_dis, min_dis = compute_euclidean_distance(labels, preds, inputs)

                    mean_dis1, dis = compute_mean_distance(labels, preds, inputs)
                    # boehler angle mean error
                    if phase == 'val':
                        angles_pred = compute_boehler_angle(input_data=preds, image=inputs)
                        angles_label = compute_boehler_angle(input_data=labels, image=inputs)
                        for t in range(len(angles_pred)):
                            angles_label_list.append(angles_label[t])
                            angles_pred_list.append(angles_pred[t])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_iter += 1
                running_loss += loss.item()
                running_mean_dis += mean_dis
                running_max_dis += max_dis
                running_min_dis += min_dis

                running_distance += mean_dis1
                running_dis1 += dis
            if phase == 'train':
                scheduler.step()

            if phase == 'val':
                # statistics 2
                angle_diff_list = np.array(angles_pred_list) - np.array(angles_label_list)
                mse_angle1, mse_angle2, angle_error_list = compute_angle_error(angles_pred=angles_pred_list,
                                                                               angles_label=angles_label_list)

            epoch_loss = running_loss/epoch_iter

            epoch_distance = running_mean_dis / epoch_iter
            dis_max = running_max_dis / epoch_iter
            dis_min = running_min_dis / epoch_iter

            epoch_distance1 = running_distance / float(epoch_iter)
            dis_1 = running_dis1 / float(epoch_iter)
            dis_1 = np.around(dis_1, decimals=1).tolist()

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print('{} Euclidean_Distance mean: {:.4f}, max: {:.4f}, min: {:.4f}'.format(
                phase, epoch_distance, dis_max, dis_min))
            if phase == 'val':
                print('{} mean_squared_angle_error: [{:.4f}, {:.4f}]'.format(
                    phase, mse_angle1, mse_angle2))

            print('{} Mean_Distance: {:.4f} {}'.format(
                phase, epoch_distance1, dis_1))
            writer.add_scalar("Loss/{}".format(phase), epoch_loss, epoch)
            writer.flush()

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_distance = epoch_distance
                best_dis_max = dis_max
                best_dis_min = dis_min
                best_mse_angle1 = mse_angle1
                best_mse_angle2 = mse_angle2
                best_angle_predictions = angles_pred_list
                best_angle_labels = angles_label_list
                best_angle_errors = np.array(angle_error_list)
                best_angle_diff = np.array(angle_diff_list)

                best_distance1 = epoch_distance1
                best_dis_1 = dis_1
        
        # visualize_model(model)  # you can activate this line if you want to see an example of how the predictions progress
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    print('Best val euclidean_distance mean: {:4f}, max: {:4f}, min: {:4f}]'.format(best_distance, best_dis_max, best_dis_min))
    print('Best val mean_squared_angle_error: [{:.4f}, {:.4f}]'.format(mse_angle1, mse_angle2))

    print('Best val mean_distance: {:4f} [{}]'.format(best_distance1, best_dis_1))


    with open('metrics.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['predictions', 'labels', 'angle_error'])  # headline
        for i in range(len(best_angle_predictions)):
            csv_writer.writerow([best_angle_predictions[i], best_angle_labels[i], best_angle_errors[i]])

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
