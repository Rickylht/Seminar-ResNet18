from training_0 import *
import torchvision.models as models
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import KFold
import pandas as pd


landmarks_frame = None
landmarks_frame = Append2Frame()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#before training, for visulization
transform_objs_basic = transforms.Compose([Contrast(),ToPILImage(),Rescale((500, 400))])
#transform_objs_basic = transforms.Compose([ToPILImage(), RandomHorizontalFlip(), Rescale((500, 400)), ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)])
basic_dataset = RadiusLandmarksDataset(data_frame = landmarks_frame, transform = transform_objs_basic, output_shape=(4,2))
def show_landmarks(image, landmarks, ax_new):
    """Show image with landmarks"""
    ax_new.imshow(image)
    ax_new.scatter(landmarks[:, 0], landmarks[:, 1], s=20, marker='.', c=['y'])
  
def preview():
  plt.figure(figsize=(10, 10))
  n_images = 3
  for i in range(3):

      n = np.random.randint(1, 200)
      n = i
      sample = basic_dataset[n]

      print(i, sample['image'].size, sample['landmarks'].size)

      ax = plt.subplot(1, n_images, i + 1)
      ax.set_title('Sample #{}'.format(n))
      ax.axis('off')
      show_landmarks(np.array(sample['image']), np.array(sample['landmarks']), ax_new=ax)

  plt.show()


def cross_validation_training():

  #transforms
  transform_objs = transforms.Compose([Contrast_Enhance(1.5, -0.5), Random_Rotate(), Addnoisy("gauss"), Random_Shift(), ToPILImage(), RandomCrop(0.6), Rescale((500, 400)), ToTensor()])
  #transform_objs = transforms.Compose([Random_Rotate(), ToPILImage(), Rescale((500, 400)), ToTensor(), Normalize()])
  transform_val = transforms.Compose([ToPILImage(), Rescale((500, 400)), ToTensor()])  # no data augmentation for the validation set (can make sense)

  # kf = KFold(n_splits = 5, random_state = None, shuffle = False)
  # for train_val_index, test_index in kf.split(landmarks_frame):
  mean_error_list = []
  i = 1

  for i in range(1,6):
    if i == 1:
      train = landmarks_frame.loc[0:120]
      val = landmarks_frame.loc[121:160]
      test = landmarks_frame.loc[161:201]
    elif i == 2:
      train = landmarks_frame.loc[41:160]
      val = landmarks_frame.loc[161:201]
      test = landmarks_frame.loc[0:40]
    elif i == 3:
      train = landmarks_frame.loc[81:201]
      val = landmarks_frame.loc[0:40]
      test = landmarks_frame.loc[41:80]
    elif i == 4:
      train = pd.concat([landmarks_frame.loc[121:201], landmarks_frame.loc[0:40]])
      val = landmarks_frame.loc[41:80]
      test = landmarks_frame.loc[81:120]
    elif i == 5:
      train = pd.concat([landmarks_frame.loc[161:201], landmarks_frame.loc[0:80]])
      val = landmarks_frame.loc[81:120]
      test = landmarks_frame.loc[121:160]

    # train_val = landmarks_frame.loc[train_val_index]
    # test = landmarks_frame.loc[test_index]
    # train, val = train_test_split(train_val, test_size=0.25, shuffle=False)
    # print(test)

    #train_val, test = train_test_split(landmarks_frame, test_size=0.2, random_state=42)
    #train, val = train_test_split(train_val, test_size=0.2, random_state=42)

    transformed_dataset_train = RadiusLandmarksDataset(data_frame=train, transform=transform_objs) #for training
    transformed_dataset_val = RadiusLandmarksDataset(data_frame=val, transform=transform_val)
    transformed_dataset_test = RadiusLandmarksDataset(data_frame=test, transform=transform_val)

    dataloader_train = DataLoader(transformed_dataset_train, batch_size=5, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(transformed_dataset_val, batch_size=5, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(transformed_dataset_test, batch_size=1, shuffle=False, num_workers=0)  

    dataloaders = {'train': dataloader_train, 'val': dataloader_val}

    #model_conv = torchvision.models.resnet34(pretrained=True)
    model_conv = torchvision.models.resnet18(pretrained=True) 
    #model_conv = torchvision.models.resnet50(pretrained=True)
    #model_conv = torchvision.models.resnet101(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = True

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = torch.nn.Linear(num_ftrs, 8)
    
    model_conv = model_conv.to(device)

    criterion = torch.nn.MSELoss()

    optimizer_conv = torch.optim.Adam(model_conv.parameters(), lr=0.001)

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_conv, step_size=10, gamma=0.1)

    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, dataloaders, device, num_epochs=30)

    #test model
    num = 0
    dis = 0.0
    for sample in dataloader_test:
      inputs = sample['image'].float().to(device)
      labels = sample['landmarks'].float().to(device)
      preds = model_conv(inputs)
      dis += ((labels-preds)**2).mean().item()
      num += 1
    avg_mean_error = dis / num
    print("Average Mean Error: ", avg_mean_error)
    mean_error_list.append(avg_mean_error)

    torch.save(model_conv, 'checkpoint_{}.pth'.format(i))

    i = i + 1
  print("Mean Error List: " , mean_error_list)


def train_best_model():
  transform_objs = transforms.Compose([Contrast(), Random_Rotate(), Addnoisy("gauss"), Random_Shift(), ToPILImage(), Rescale((500, 400)), ToTensor()])
  transform_val = transforms.Compose([ToPILImage(), Rescale((500, 400)), ToTensor()])

  train, val = train_test_split(landmarks_frame, test_size=0.2, shuffle=False)
  transformed_dataset_train = RadiusLandmarksDataset(data_frame=landmarks_frame, transform=transform_objs) #all datas for training
  transformed_dataset_val = RadiusLandmarksDataset(data_frame=val, transform=transform_val)

  dataloader_train = DataLoader(transformed_dataset_train, batch_size=10, shuffle=True, num_workers=0)
  dataloader_val = DataLoader(transformed_dataset_val, batch_size=10, shuffle=True, num_workers=0)

  dataloaders = {'train': dataloader_train, 'val': dataloader_val}

  #marked
  model_conv = torchvision.models.resnet18(pretrained=True)
  for param in model_conv.parameters():
      param.requires_grad = True

  num_ftrs = model_conv.fc.in_features
  model_conv.fc = torch.nn.Linear(num_ftrs, 8)
  
  model_conv = model_conv.to(device)

  criterion = torch.nn.MSELoss()

  optimizer_conv = torch.optim.Adam(model_conv.parameters(), lr=0.001)

  exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_conv, step_size=10, gamma=0.1)

  model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, dataloaders, device, num_epochs=30)

  torch.save(model_conv, 'checkpoint_best.pth')

  return


#visualize loss of data
def show_landmarks_2(image, landmarks_gt, landmarks_pred, ax_new):
  """Show image with landmarks"""
  ax_new.imshow(image)
  ax_new.scatter(landmarks_gt[:, 0], landmarks_gt[:, 1], s=20, marker='.', c='g')     #labelled, in green
  ax_new.scatter(landmarks_pred[:, 0], landmarks_pred[:, 1], s=20, marker='.', c='r') #predicted, in red


transform = transforms.Compose([ToPILImage(), Rescale((500, 400)), ToTensor()])
transformed_dataset = RadiusLandmarksDataset(data_frame=landmarks_frame, transform=transform)
dataloader = DataLoader(transformed_dataset, batch_size=10, shuffle=True, num_workers=0)


def visualize_model(dataloader):
  last_activation = torch.nn.Identity()
  model = torch.load("checkpoint_best.pth")
  was_training = model.training
  images_so_far = 0
  num_images = 5
  fig = plt.figure(figsize=(10, 10))

  with torch.no_grad():
    for sample in dataloader:
      inputs = sample['image'].float().to(device)
      labels = sample['landmarks'].float().to(device)
      outputs = model(inputs)
      preds = last_activation(outputs)

      for j in range(inputs.size()[0]):
        lms = preds.cpu().data[j].reshape(4, 2)
        lms_gt = labels.cpu().data[j].reshape(4, 2)
        images_so_far += 1
        ax = plt.subplot(1, num_images, images_so_far)
        ax.set_title('Loss {:.3f}'.format(((lms_gt-lms)**2).mean().item()))
        ax.axis('off')
        img = inputs.cpu().data[j].permute(1, 2, 0)
        show_landmarks_2(np.array(img), np.array(lms_gt)*np.array(img.shape[:2]), np.array(lms)*np.array(img.shape[:2]), ax_new=ax)
        plt.tight_layout()
        if images_so_far == num_images:
            model.train(mode=was_training)
            plt.show()
            return
    model.train(mode=was_training)
    plt.show()


#test of single image
def test_single_image(path = "image4.png"):
  model = torch.load("checkpoint_best.pth")

  image_name = os.path.join(path)
  image = io.imread(image_name)
  transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((500,400)), transforms.ToTensor()])

  inputs = transform(image)
  inputs = inputs.cuda()
  outputs = model(inputs.unsqueeze(0))
  point_list = outputs.tolist()[0]
  point_x = point_list[::2]
  point_x_new = [i * 500 for i in point_x]
  point_y = point_list[1::2]
  point_y_new = [i * 400 for i in point_y]
  plt.figure("Prediction")
  plt.imshow(io.resize(image, (400,500), interpolation= io.INTER_AREA))
  plt.title('Prediction')
  plt.scatter(point_x_new, point_y_new, marker = '.', c = ['g','g','y','y'])
  plt.show()


#for GUI implementation
def get_results(path = "image4.png"):    #return image, bohler angle, status
  model = torch.load("checkpoint_best.pth") 
  
  image_name = os.path.join(path)
  image = io.imread(image_name)

  transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((500,400)), transforms.ToTensor()])

  inputs = transform(image)
  inputs = inputs.cuda()
  outputs = model(inputs.unsqueeze(0))
  point_list = outputs.tolist()[0]
  point_x = point_list[::2]
  point_x_new = [i * 500 for i in point_x]
  point_x_new = list(map(int, point_x_new)) #transfer to int
  point_y = point_list[1::2]
  point_y_new = [i * 400 for i in point_y]
  point_y_new = list(map(int, point_y_new))
  image = io.resize(image, (400,500), interpolation= io.INTER_AREA)

  #draw lines
  io.line(image, (point_x_new[0], point_y_new[0]), (point_x_new[1], point_y_new[1]), (0, 0, 255), 1, 4)
  io.line(image, (point_x_new[2], point_y_new[2]), (point_x_new[3], point_y_new[3]), (255, 0, 0), 1, 4)
  io.imwrite('./images/image_contour.png', image)
  #plt.imshow(image)
  #plt.show()

  #compute angle
  AB = [point_x_new[0], point_y_new[0], point_x_new[1], point_y_new[1]]
  CD = [point_x_new[2], point_y_new[2], point_x_new[3], point_y_new[3]]

  def compute_angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

  angle = compute_angle(AB, CD)
  angle = abs(90 - angle)
  #print("angle =", angle)
  
  #decide status
  status = "Lateral"

  return angle, status
  

#1. run this to get a preview of input data
#preview()

#2. run this to train and save 5 model(in total about 12 mins)
#cross_validation_training()

#3. choose best model and train again
#train_best_model()

#4. run this to visualize the loss from validation data
#visualize_model(dataloader)

#5. run this to test single extra image
#test_single_image()



