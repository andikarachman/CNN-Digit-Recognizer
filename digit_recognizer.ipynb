{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Digit Recognizer Using Convolutional Neural Networks\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![title](assets/mnist.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this notebook, we will identify digits from a dataset of tens of thousands of handwritten images, by utilizing convolutional neural networks (CNN). [MNIST handwritten digit dataset](https://www.kaggle.com/c/digit-recognizer/data) is used to train and test the CNN model.\n",
    "\n",
    "We break this notebook into separate steps.  \n",
    "\n",
    "* [Step 1](#step1): Import Datasets\n",
    "* [Step 2](#step2): Specify Data Loaders for the Image Dataset\n",
    "* [Step 3](#step3): Define Model Architecture\n",
    "* [Step 4](#step4): Specify Loss Function and Optimizer\n",
    "* [Step 5](#step5): Train and Validate the Model\n",
    "* [Step 6](#step6): Test the Model\n",
    "\n",
    "Before moving to the next section, we need to import all packages required to do the analysis by calling the following:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Data analysis packages\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Data visualizaiton packages\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline \n",
    "\n",
    "# Deep learning packages\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "import torch.nn.functional as F\n",
    "import torch.utils.data as data_utils\n",
    "import torchvision.transforms as transforms\n",
    "from torchvision import datasets"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a id='step1'></a>\n",
    "## Step 1: Import Datasets\n",
    "[MNIST handwritten digit dataset](https://www.kaggle.com/c/digit-recognizer/data) is used to train and test the CNN model. We also perform a grayscale normalization to reduce the effect of illumination's differences."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Image shape: (42000, 784)\n",
      "Labels shape: (42000,)\n"
     ]
    }
   ],
   "source": [
    "# Import datasets\n",
    "df = pd.read_csv('data/train.csv', dtype=np.float32)\n",
    "labels = df['label'].values\n",
    "img = df.drop(labels='label', axis=1).values / 255 # Normalization\n",
    "\n",
    "# Show the shape of the dataset\n",
    "print(\"Image shape: {}\".format(img.shape))\n",
    "print(\"Labels shape: {}\".format(labels.shape))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a id='step2'></a>\n",
    "## Step 2: Specify Data Loaders for the Image Dataset\n",
    "The images (28px x 28px) has been stocked into pandas dataframe as 1D vectors of 784 values. We reshape all data to 28x28 matrices. PyTorch requires an extra dimension in the beginning, which correspond to channels. MNIST images are gray scaled so it use only one channel. For RGB images, there is 3 channels, we would have reshaped 784px vectors to 3x28x28 3D matrices."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Reshape images\n",
    "img = img.reshape(-1, 1, 28, 28)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Then, we split the train set in three parts : \n",
    "- 60% of the dataset become the train set\n",
    "- 20% of the dataset become the validation set\n",
    "- 20% of the dataset become the test set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Split dataset into train and validation/test set\n",
    "img_train, img_val_test, labels_train, labels_val_test = train_test_split(img, \n",
    "                                                                          labels, \n",
    "                                                                          test_size=0.4, \n",
    "                                                                          random_state=42,\n",
    "                                                                          stratify=labels)\n",
    "\n",
    "# Split validation/test set into validation and test set\n",
    "img_val, img_test, labels_val, labels_test = train_test_split(img_val_test, \n",
    "                                                              labels_val_test, \n",
    "                                                              test_size=0.5, \n",
    "                                                              random_state=42,\n",
    "                                                              stratify=labels_val_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We use the code cell below to write three separate [data loaders](http://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of digit images. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Define batch_size, epoch and iteration\n",
    "batch_size = 100\n",
    "n_iters = 2000\n",
    "num_epochs = n_iters / (len(img_train) / batch_size)\n",
    "num_epochs = int(num_epochs)\n",
    "\n",
    "#---------------------------------------------------------------------------------------------------------------\n",
    "\n",
    "# Convert train set to tensors\n",
    "img_train = torch.from_numpy(img_train)\n",
    "labels_train = torch.from_numpy(labels_train).type(torch.LongTensor)\n",
    "\n",
    "# Convert validation set to tensors\n",
    "img_val = torch.from_numpy(img_val)\n",
    "labels_val = torch.from_numpy(labels_val).type(torch.LongTensor)\n",
    "\n",
    "# Convert test set to tensors\n",
    "img_test = torch.from_numpy(img_test)\n",
    "labels_test = torch.from_numpy(labels_test).type(torch.LongTensor)\n",
    "\n",
    "#---------------------------------------------------------------------------------------------------------------\n",
    "\n",
    "# Define Pytorch train and validation set\n",
    "train = data_utils.TensorDataset(img_train, labels_train)\n",
    "val = data_utils.TensorDataset(img_val, labels_val)\n",
    "test = data_utils.TensorDataset(img_test, labels_test)\n",
    "\n",
    "#---------------------------------------------------------------------------------------------------------------\n",
    "\n",
    "# Define data loader\n",
    "train_loader = data_utils.DataLoader(train, \n",
    "                                     batch_size=batch_size, \n",
    "                                     shuffle=True, num_workers=16)\n",
    "valid_loader = data_utils.DataLoader(val, \n",
    "                                     batch_size=batch_size, \n",
    "                                     shuffle=True, num_workers=16)\n",
    "test_loader = data_utils.DataLoader(test, \n",
    "                                    batch_size=batch_size, \n",
    "                                    shuffle=True, num_workers=16)\n",
    "\n",
    "loaders = {'train': train_loader,\n",
    "           'valid': valid_loader,\n",
    "           'test': test_loader}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a id='step3'></a>\n",
    "## Step 3: Define Model Architecture\n",
    "In our CNN architecture, the first layer has input shape of (28, 28, 1) and the last layer should output 10 classes. In the first two convolutional layers, we choose to set 32 filters with `kernel_size` of 5. After these layers, the data is downsampled by using a max pooling layer with stride of 1. The next two convolutional layers have 64 filters with `kernel_size` of 3. Then, the data is downsampled by using a max pooling layer with stride of 2.\n",
    "\n",
    "We have applied dropout of 0.5. Dropout is a regularization method, where a proportion of nodes in the layer are randomly ignored (setting their wieghts to zero) for each training sample. This drops randomly a propotion of the network and forces the network to learn features in a distributed way. This technique also improves generalization and reduces the overfitting. \n",
    "\n",
    "We use 'ReLu' as our activation function. ReLu is used to add non linearity to the network.\n",
    "\n",
    "Fully-connected layer is placed at the end of the network. It combines all the found local features of the previous convolutional layers. Then, the 2nd fully-connected layer is intended to produce final output size, which predicts classes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Net(\n",
      "  (conv1): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))\n",
      "  (conv2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))\n",
      "  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "  (conv4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "  (pool1): MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)\n",
      "  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
      "  (fc1): Linear(in_features=7744, out_features=2048, bias=True)\n",
      "  (fc2): Linear(in_features=2048, out_features=10, bias=True)\n",
      "  (dropout): Dropout(p=0.5)\n",
      ")\n"
     ]
    }
   ],
   "source": [
    "num_classes = 10\n",
    "\n",
    "# check if CUDA is available\n",
    "use_cuda = torch.cuda.is_available()\n",
    "\n",
    "# Define the CNN architecture\n",
    "class Net(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(Net, self).__init__()\n",
    "        ## Define layers of a CNN\n",
    "        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=1)\n",
    "        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=1)\n",
    "        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)\n",
    "        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)\n",
    "        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)\n",
    "        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) \n",
    "        self.fc1 = nn.Linear(64 * 11 * 11, 2048)\n",
    "        self.fc2 = nn.Linear(2048, num_classes) \n",
    "        self.dropout = nn.Dropout(0.5)\n",
    "    \n",
    "    def forward(self, x):\n",
    "        ## Define forward behavior\n",
    "        x = self.conv1(x)\n",
    "        x = self.pool1(F.relu(self.conv2(x)))\n",
    "        x = self.dropout(x)\n",
    "        x = self.conv3(x)\n",
    "        x = self.pool2(F.relu(self.conv4(x)))\n",
    "        x = self.dropout(x)\n",
    "        #print(x.shape)\n",
    "        x = x.view(-1, 64 * 11 * 11)\n",
    "        x = self.dropout(x)\n",
    "        x = F.relu(self.fc1(x))\n",
    "        x = self.dropout(x)\n",
    "        x = self.fc2(x)\n",
    "        return x\n",
    "\n",
    "# instantiate the CNN\n",
    "model = Net()\n",
    "\n",
    "# move tensors to GPU if CUDA is available\n",
    "if use_cuda:\n",
    "    model.cuda()\n",
    "\n",
    "print(model)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a id='step4'></a>\n",
    "## Step 4: Specify Loss Function and Optimizer\n",
    "We use `CrossEntropyLoss` as our loss function and `RMSprop` as our optimizer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "### Define loss function\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "\n",
    "### Define optimizer\n",
    "optimizer = optim.RMSprop(model.parameters(), \n",
    "                                  lr=0.001, \n",
    "                                  alpha=0.9, \n",
    "                                  eps=1e-08, \n",
    "                                  weight_decay=0.0001)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a id='step5'></a>\n",
    "## Step 5: Train and Validate the Model\n",
    "We [save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'cnn_digit_recognizer.pt'`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# The following import is required for training to be robust to truncated images\n",
    "from PIL import ImageFile\n",
    "ImageFile.LOAD_TRUNCATED_IMAGES = True\n",
    "\n",
    "def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):\n",
    "    \"\"\"returns trained model\"\"\"\n",
    "    # initialize tracker for minimum validation loss\n",
    "    valid_loss_min = np.Inf\n",
    "    loss_list = []\n",
    "    epoch_list = []\n",
    "    \n",
    "    for epoch in range(1, n_epochs+1):\n",
    "        # initialize variables to monitor training and validation loss\n",
    "        train_loss = 0.0\n",
    "        valid_loss = 0.0\n",
    "        \n",
    "        ###################\n",
    "        # train the model #\n",
    "        ###################\n",
    "        model.train()\n",
    "        for batch_idx, (data, target) in enumerate(loaders['train']):\n",
    "            # move to GPU\n",
    "            if use_cuda:\n",
    "                data, target = data.cuda(), target.cuda()\n",
    "            ## find the loss and update the model parameters accordingly\n",
    "            optimizer.zero_grad()\n",
    "            output = model(data)\n",
    "            loss = criterion(output, target)\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))\n",
    "            \n",
    "        ######################    \n",
    "        # validate the model #\n",
    "        ######################\n",
    "        model.eval()\n",
    "        for batch_idx, (data, target) in enumerate(loaders['valid']):\n",
    "            # move to GPU\n",
    "            if use_cuda:\n",
    "                data, target = data.cuda(), target.cuda()\n",
    "            ## update the average validation loss\n",
    "            output = model(data)\n",
    "            loss = criterion(output, target)\n",
    "            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))\n",
    "\n",
    "            \n",
    "        # print training/validation statistics \n",
    "        print('Epoch: {} \\tTraining Loss: {:.6f} \\tValidation Loss: {:.6f}'.format(\n",
    "            epoch, \n",
    "            train_loss,\n",
    "            valid_loss\n",
    "            ))\n",
    "        \n",
    "        ## Save the model if validation loss has decreased\n",
    "        if valid_loss < valid_loss_min:\n",
    "            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))\n",
    "            torch.save(model.state_dict(), save_path)\n",
    "            valid_loss_min = valid_loss\n",
    "            \n",
    "    # return trained model\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 1 \tTraining Loss: 0.385877 \tValidation Loss: 0.121577\n",
      "Validation loss decreased (inf --> 0.121577).  Saving model ...\n",
      "Epoch: 2 \tTraining Loss: 0.139635 \tValidation Loss: 0.070934\n",
      "Validation loss decreased (0.121577 --> 0.070934).  Saving model ...\n",
      "Epoch: 3 \tTraining Loss: 0.122056 \tValidation Loss: 0.074650\n",
      "Epoch: 4 \tTraining Loss: 0.117830 \tValidation Loss: 0.089532\n",
      "Epoch: 5 \tTraining Loss: 0.118097 \tValidation Loss: 0.069858\n",
      "Validation loss decreased (0.070934 --> 0.069858).  Saving model ...\n",
      "Epoch: 6 \tTraining Loss: 0.111873 \tValidation Loss: 0.124239\n",
      "Epoch: 7 \tTraining Loss: 0.110335 \tValidation Loss: 0.067964\n",
      "Validation loss decreased (0.069858 --> 0.067964).  Saving model ...\n",
      "Epoch: 8 \tTraining Loss: 0.102659 \tValidation Loss: 0.072183\n",
      "Epoch: 9 \tTraining Loss: 0.107836 \tValidation Loss: 0.070678\n",
      "Epoch: 10 \tTraining Loss: 0.104341 \tValidation Loss: 0.065509\n",
      "Validation loss decreased (0.067964 --> 0.065509).  Saving model ...\n",
      "Epoch: 11 \tTraining Loss: 0.104885 \tValidation Loss: 0.084792\n",
      "Epoch: 12 \tTraining Loss: 0.104520 \tValidation Loss: 0.054289\n",
      "Validation loss decreased (0.065509 --> 0.054289).  Saving model ...\n",
      "Epoch: 13 \tTraining Loss: 0.106377 \tValidation Loss: 0.067225\n",
      "Epoch: 14 \tTraining Loss: 0.093840 \tValidation Loss: 0.054075\n",
      "Validation loss decreased (0.054289 --> 0.054075).  Saving model ...\n",
      "Epoch: 15 \tTraining Loss: 0.103423 \tValidation Loss: 0.091325\n",
      "Epoch: 16 \tTraining Loss: 0.096374 \tValidation Loss: 0.059280\n",
      "Epoch: 17 \tTraining Loss: 0.097545 \tValidation Loss: 0.087232\n",
      "Epoch: 18 \tTraining Loss: 0.096278 \tValidation Loss: 0.115145\n",
      "Epoch: 19 \tTraining Loss: 0.092485 \tValidation Loss: 0.055406\n",
      "Epoch: 20 \tTraining Loss: 0.096590 \tValidation Loss: 0.080501\n",
      "Epoch: 21 \tTraining Loss: 0.093170 \tValidation Loss: 0.046434\n",
      "Validation loss decreased (0.054075 --> 0.046434).  Saving model ...\n",
      "Epoch: 22 \tTraining Loss: 0.098416 \tValidation Loss: 0.060400\n",
      "Epoch: 23 \tTraining Loss: 0.092794 \tValidation Loss: 0.078424\n",
      "Epoch: 24 \tTraining Loss: 0.088741 \tValidation Loss: 0.128129\n",
      "Epoch: 25 \tTraining Loss: 0.090808 \tValidation Loss: 0.094697\n",
      "Epoch: 26 \tTraining Loss: 0.085207 \tValidation Loss: 0.066789\n",
      "Epoch: 27 \tTraining Loss: 0.091125 \tValidation Loss: 0.079907\n",
      "Epoch: 28 \tTraining Loss: 0.088407 \tValidation Loss: 0.071867\n",
      "Epoch: 29 \tTraining Loss: 0.095062 \tValidation Loss: 0.067375\n",
      "Epoch: 30 \tTraining Loss: 0.091764 \tValidation Loss: 0.061416\n"
     ]
    }
   ],
   "source": [
    "# Train the model\n",
    "model = train(30, loaders, model, optimizer, \n",
    "              criterion, use_cuda, 'cnn_digit_recognizer.pt')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a id='step6'></a>\n",
    "## Step 6: Test the Model\n",
    "We try out your model on the test dataset.  We use the code cell below to calculate and print the test loss and accuracy."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "IncompatibleKeys(missing_keys=[], unexpected_keys=[])"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Load the model that got the best validation accuracy\n",
    "model.load_state_dict(torch.load('cnn_digit_recognizer.pt'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test Loss: 0.041995\n",
      "\n",
      "\n",
      "Test Accuracy: 98% (8301/8400)\n"
     ]
    }
   ],
   "source": [
    "def test(loaders, model, criterion, use_cuda):\n",
    "\n",
    "    # monitor test loss and accuracy\n",
    "    test_loss = 0.\n",
    "    correct = 0.\n",
    "    total = 0.\n",
    "\n",
    "    model.eval()\n",
    "    for batch_idx, (data, target) in enumerate(loaders['test']):\n",
    "        # move to GPU\n",
    "        if use_cuda:\n",
    "            data, target = data.cuda(), target.cuda()\n",
    "        # forward pass: compute predicted outputs by passing inputs to the model\n",
    "        output = model(data)\n",
    "        # calculate the loss\n",
    "        loss = criterion(output, target)\n",
    "        # update average test loss \n",
    "        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))\n",
    "        # convert output probabilities to predicted class\n",
    "        pred = output.data.max(1, keepdim=True)[1]\n",
    "        # compare predictions to true label\n",
    "        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())\n",
    "        total += data.size(0)\n",
    "            \n",
    "    print('Test Loss: {:.6f}\\n'.format(test_loss))\n",
    "\n",
    "    print('\\nTest Accuracy: %2d%% (%2d/%2d)' % (\n",
    "        100. * correct / total, correct, total))\n",
    "\n",
    "# call test function    \n",
    "test(loaders, model, criterion, use_cuda)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Visualize Sample Test Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# helper function to un-normalize and display an image\n",
    "def imshow(img):\n",
    "    img = img.numpy() * 255  # unnormalize and convert from Tensor image\n",
    "    plt.imshow(img[0])  # show image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x360 with 10 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "classes = ['0', '1', '2', '3', '4',\n",
    "           '5', '6', '7', '8', '9']\n",
    "\n",
    "# obtain one batch of test images\n",
    "dataiter = iter(test_loader)\n",
    "images, labels = dataiter.next()\n",
    "images.numpy()\n",
    "\n",
    "# move model inputs to cuda, if GPU available\n",
    "if use_cuda:\n",
    "    images = images.cuda()\n",
    "\n",
    "# get sample outputs\n",
    "output = model(images)\n",
    "# convert output probabilities to predicted class\n",
    "_, preds_tensor = torch.max(output, 1)\n",
    "preds = np.squeeze(preds_tensor.numpy()) if not use_cuda else np.squeeze(preds_tensor.cpu().numpy())\n",
    "\n",
    "# plot the images in the batch, along with predicted and true labels\n",
    "fig = plt.figure(figsize=(20, 5))\n",
    "for idx in np.arange(10):\n",
    "    ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])\n",
    "    imshow(images.cpu()[idx])\n",
    "    ax.set_title(\"Predicted: {} - Actual: {}\".format(classes[preds[idx]], classes[labels[idx]]),\n",
    "                 color=(\"green\" if preds[idx]==labels[idx].item() else \"red\"))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
