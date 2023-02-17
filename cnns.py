import torch.nn as nn

class CNN_1(nn.Module):
    """
    Creation of the neural network
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
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(25*25*25, 10)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(10, 3)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_data):
        """
        Order of the layers
        :param input_data: Input image
        :return: a tensor of size (1, number of classes) (Softmax)
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
    Creation of the neural network
    """

    def __init__(self, dim_in):
        """
        Initialisation of the layers
        :param dim_in: dimension of the input image
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=dim_in, out_channels=10, kernel_size=(5, 5),stride=(3,3))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3),stride=(2,2),padding=(2,2))
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(3, 3),stride=(2, 2),padding=(2,2))

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(50*3*3, 30)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(30, 3)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_data):
        """
        Order of the layers
        :param input_data: Input image
        :return: a tensor of size (1, number of classes) (Softmax)
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