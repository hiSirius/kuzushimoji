
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
from PIL import Image
import numpy as np
import os
import csv


class Net( nn.Module ):
    def __init__( self, outputFeatures=48 ):
        super( Net, self ).__init__()

        # download pretrained model into "C:\Users\[UserName]\.cache\torch\checkpoints"
        model = models.resnet50( pretrained=True ) # resnet32 resnet18

        # if you want to use VGG model
        # model = models.vgg11( pretrained=True ) # vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn

        # fine tuning1
        self.conv1 = nn.Conv2d( in_channels=1, out_channels=model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride, padding=model.conv1.padding, bias=model.conv1.bias )
        self.model = nn.Sequential( *list( model.children() )[1:-2] )
        self.maxpool = nn.MaxPool2d( kernel_size=7 )
        self.fc = nn.Linear( model.fc.in_features, outputFeatures )

    def forward( self, x ):
        x = self.conv1( x )
        x = self.model( x )
        x = -self.maxpool( -x )
        x = x.view( -1, self.fc.in_features )
        x = self.fc( x )
        return x


class KanaDataset( torch.utils.data.Dataset ):
    def __init__( self, rootFolder, transform=None, isTest=False ):
        self.rootFolder = os.path.abspath( rootFolder )
        self.classes = sorted( os.listdir( self.rootFolder ) )
        self.transform = transform
        self.datas = []
        self.labels = []
        for i, kana in enumerate( self.classes, 0 ):
            kanaFolder = os.path.join( self.rootFolder, kana )
            images = [file for file in sorted( os.listdir( kanaFolder ) ) if ".jpg" in file]

            count = 0
            for j, image in enumerate( images, 0 ):
                if not image.startswith( '.' ):# and j % 32 == 0:
                    if count % 5 == 0:
                        if isTest:
                            self.datas.append( image )
                            self.labels.append( i )
                    else:
                        if not isTest:
                            self.datas.append( image )
                            self.labels.append( i )
                    count += 1

        self.len = len( self.datas )


    def __len__( self ):
        return self.len

    def __getitem__( self, idx ):
        # generate data
        imageName = os.path.join( self.rootFolder, self.classes[ self.labels[ idx ] ] + '/' + self.datas[ idx ] )
        """
        # extract character area using binary
        image = cv2.imread( imageName )
        thresh, binImage = cv2.threshold( cv2.cvtColor( image, cv2.COLOR_BGR2GRAY ), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
        binImage = cv2.cvtColor( binImage, cv2.COLOR_GRAY2BGR )
        background = np.ones( binImage.shape[:3], np.uint8 ) * 255
        imageHeight, imageWidth = binImage.shape[:2]
        outData = np.where( binImage == 0, image, background[ 0:imageHeight, 0:imageWidth, : ] )
        """
        outData = Image.fromarray( cv2.imread( imageName, cv2.IMREAD_GRAYSCALE ) )

        if self.transform:
            outData = self.transform( outData )

        # generate label
        outLabel = self.labels[ idx ]

        return outData, outLabel

    def getClasses( self ):
        return self.classes
