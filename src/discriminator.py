
import csv
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torchvision.transforms as transforms

import resnet
import split
import utils

def split_multi(imagePath):
    return split.kmeans_black(imagePath) + split.kmeans_black(imagePath, k=4)
    #return split.hist(imagePath) + split.hist(imagePath, k=4)

def discriminate():
    #rootPath = "./test/"
    rootPath = "C:/Users/Nakaizumi/Desktop/2019camp/test1/"
    imagesPath = os.path.join( rootPath, "imgs" )
    annotationPath = os.path.join( rootPath, "annotations.csv" )
    resultPath = os.path.join( rootPath, "result.csv" )

    #modelPath = "./result/30_weight.pth"
    kanaFolder = "C:/Users/Nakaizumi/Desktop/2019camp/alcon2019_tar/alcon2019/alcon2019/dataset/train_kana"
    #modelPath = "C:/Users/Nakaizumi/Desktop/2019camp/sakamura/trained-model/epoch_16_resnet50/30_weight.pth"
    modelPath = "C:/Users/Nakaizumi/Desktop/2019camp/sakamura/final/122_weight.pth"

    #kanaFolder = "./train_kana/"
    classes = sorted( os.listdir( os.path.abspath( kanaFolder ) ) )

    pool = mp.Pool(mp.cpu_count() // 2)

    # load image name and annotations from csv file
    imageNames = []
    annotations = []
    with open( annotationPath, 'r' ) as f:
        reader = csv.reader( f )
        header = next( reader )
        for row in reader:
            imageNames.append( row[0] )
            annotations.append( row[1:] )

    transform = transforms.Compose( [ transforms.Resize( ( 224, 224 ) ), transforms.ToTensor(), transforms.Normalize( [ 0.5 ], [ 0.25 ] ) ] )

    # use CUDA
    device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
    print( device )

    # generate net
    net = resnet.Net( outputFeatures=len( classes ) )
    net.load_state_dict( torch.load( modelPath ) )
    net = net.to( device )

    # calcurate elaped time
    starttime = time.time()
    print( "Start Discrimination" )

    # write result into csv file
    with open( resultPath, 'w' ) as f:
        writer = csv.writer( f, lineterminator='\n' )
        writer.writerow( header )

        # discrimination
        with torch.no_grad():
            args = [os.path.join( imagesPath, imageName + ".jpg" )
                for imageName in imageNames]

            for imageName, images in zip(imageNames, pool.imap(split_multi, args)):
                inputs = torch.cat( [ transform( image ).unsqueeze( 0 ) for image in images ], dim=0 )
                inputs = inputs.to( device )

                outputs = net( inputs )
                prob, predicted = torch.max( outputs, 1 )


                if torch.sum(prob[:3]) < torch.sum(prob[3:] - torch.min(prob[3:])):
                    print("selected 4 class", prob, torch.sum(prob[:3]), "<", torch.sum(prob[3:] - torch.min(prob[3:])))
                    predicted = [ classes[ p ] for i, p in enumerate(predicted[3:]) if i != torch.argmin(prob[3:])]
                else:
                    print("selected 3 class", prob, torch.sum(prob[:3]), ">", torch.sum(prob[3:] - torch.min(prob[3:])))
                    predicted = [ classes[ p ] for p in predicted[:3] ]

                writer.writerow( [ imageName ] + predicted )
               
    elapsed = time.time() - starttime
    print( "Discrimination Finished [%d sec]" % elapsed )
    return

def evaluate():
    #rootPath = "./test/"
    rootPath = "C:/Users/Nakaizumi/Desktop/2019camp/test/"
    annotationPath = os.path.join( rootPath, "annotations.csv" )
    resultPath = os.path.join( rootPath, "result.csv" )

    # ground truth
    annotations = []
    with open( annotationPath, 'r' ) as f:
        reader = csv.reader( f )
        for row in reader:
            annotations.append( row[1:] )

    # predicted results
    results = []
    with open( resultPath, 'r' ) as f:
        reader = csv.reader( f )
        for row in reader:
            results.append( row[1:] )

    # check same sizes between two lists
    assert len( annotations ) == len( results )

    eval = [0] * 4
    for annotation, result in zip( annotations, results ):
        num = [ True if a == r else False for a, r in zip( annotation, result ) ].count( True )
        eval[ num ] += 1

    print( "The number of correct characters" )
    for i, e in enumerate( eval, 0 ):
        print( "  %d : %6d images" % ( i, e ) )

    total = sum( eval )
    accuracy = 100 * ( eval[1] + 2 * eval[2] + 3 * eval[3] ) / ( 3 * total )
    score = 100 * eval[3] / total

    print( "Accuracy of a character = %d %%" % accuracy )
    print( "Score = %d %%" % score )
    return

if __name__ == "__main__":
    discriminate()
    #evaluate()
