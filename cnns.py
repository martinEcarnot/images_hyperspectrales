import torch.nn as nn

class CNN_1(nn.Module):
    """
    Personal class of a light convolutional neural network
    """

    def __init__(self, dim_in):
        """
        Initialisation of the layers
        :param dim_in: dimension of the input image
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=dim_in, out_channels=10, kernel_size=(7, 7),stride=(1,1),padding=(3,3))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=(5, 5),stride=(1,1),padding=(2,2))
        self.conv3 = nn.Conv2d(in_channels=15, out_channels=25, kernel_size=(3, 3),stride=(1, 1),padding=(1,1))

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(25*25*25, 10)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(10, 2)

        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_data):
        """
        Performs the forward pass of an input vector
        :param input_data: Input image
        :return: a tensor of size (1, number of classes) containing the probabilities associated with each class.
        """
        x = self.pool(self.relu(self.conv1(input_data)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

    
    
class CNN_2(nn.Module):
    """
    Personal class of a medium convolutional neural network
    """

    def __init__(self, dim_in):
        """
        Initialisation of the layers
        :param dim_in: dimension of the input image
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=dim_in, out_channels=8, kernel_size=(7, 7),padding=(2, 2),stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5),stride=(1, 1),padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3),stride=(1, 1),padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),stride=(1, 1),padding=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),stride=(1, 1),padding=(1, 1))


        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(4*4*128, 100)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(100, 20)
        self.linear3 = nn.Linear(20, 2)

        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_data):
        """
        Performs the forward pass of an input vector
        :param input_data: Input image
        :return: a tensor of size (1, number of classes) containing the probabilities associated with each class.
        """
        x = self.pool1(self.relu(self.conv1(input_data)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool2(self.relu(self.conv3(x)))
        x = self.pool2(self.relu(self.conv4(x)))
        x = self.pool2(self.relu(self.conv5(x)))
        x = self.flatten(x)
        

        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)

        x = self.relu(x)
        x = self.linear3(x)


        x = self.softmax(x)
        return x
    

class CNN_3(nn.Module):
    """
    Personal class of a larger convolutional neural network
    """

    def __init__(self, dim_in):
        """
        Initialisation of the layers
        :param dim_in: dimension of the input image
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=dim_in, out_channels=32, kernel_size=(7, 7),padding=(2, 2),stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5),stride=(1, 1),padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),stride=(1, 1),padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),stride=(1, 1),padding=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),stride=(1, 1),padding=(1, 1))


        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(4*4*256, 100)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(100, 30)
        self.linear3 = nn.Linear(30, 2)

        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_data):
        """
        Performs the forward pass of an input vector
        :param input_data: Input image
        :return: a tensor of size (1, number of classes) containing the probabilities associated with each class.
        """
        x = self.pool1(self.relu(self.conv1(input_data)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool2(self.relu(self.conv3(x)))
        x = self.pool2(self.relu(self.conv4(x)))
        x = self.pool2(self.relu(self.conv5(x)))
        x = self.flatten(x)
        

        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)

        x = self.relu(x)
        x = self.linear3(x)


        x = self.softmax(x)
        return x


