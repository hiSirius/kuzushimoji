
import os
import csv
import time
import cv2
import multiprocessing as multi
from multiprocessing import Pool
import numpy as np


def matchTemplate( kana, kanaDirectory, kanasImage, kanasW, kanasH, method ):
    kanaImage = cv2.imread( os.path.join( kanaDirectory, kana ), 0 )
    kanaW, kanaH = kanaImage.shape[::-1]

    if kanaW > kanasW or kanaH > kanasH:
        return -1.0, [ 0, 0 ], kanaW, kanaH

    res = cv2.matchTemplate( kanasImage, kanaImage, method )
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc( res )
    return maxVal, maxLoc, kanaW, kanaH


def wrapper( args ):
    return matchTemplate( *args )


def extend():
    rootPath = "D:/ImageLab/Camp/2019/alcon2019/dataset/"

    kanasPath = os.path.join( rootPath, "train/imgs" )
    kanaPath = os.path.join( rootPath, "train_kana" )
    annotationPath = os.path.join( rootPath, "train/annotations.csv" )
    resultPath = os.path.join( rootPath, "train/annotations_ex.csv" )

    # load image name and annotations from csv file
    imageNames = []
    annotations = []
    header = None
    with open( annotationPath, 'r' ) as f:
        reader = csv.reader( f )
        header = next( reader )
        for row in reader:
            imageNames.append( row[0] )
            annotations.append( row[1:] )

    # set evaluation method for template matching
    method = eval( "cv2.TM_CCOEFF_NORMED" )

    extendHeader = [ "TopLeftX1", "TopLeftY1", "BottomRightX1", "BottomRightY1", "Likeliood1",
                     "TopLeftX2", "TopLeftY2", "BottomRightX2", "BottomRightY2", "Likeliood2",
                     "TopLeftX3", "TopLeftY3", "BottomRightX3", "BottomRightY3", "Likeliood3" ]

    # multiprocess for template matching
    p = Pool( multi.cpu_count() )

    # calcurate elapsed time
    starttime = time.time()
    sectiontime = starttime
    print( "Start Extension" )

    outlier = 0

    # write result into csv file
    with open( resultPath, 'w' ) as f:
        writer = csv.writer( f, lineterminator='\n' )
        writer.writerow( header + extendHeader )

        # open three kana image
        for i, imageName in enumerate( imageNames, 0 ):
            kanasName = os.path.join( kanasPath, imageName + ".jpg" )
            kanasImage = cv2.imread( kanasName, 0 )
            kanasW, kanasH = kanasImage.shape[::-1]

            additionalDatas = []

            # open annottion file
            for annotation in annotations[i]:
                kanaDirectory = os.path.join( kanaPath, annotation )
                kanaFiles = os.listdir( os.path.abspath( kanaDirectory ) )

                kanaMaxVal = -1.0
                kanaMaxName = None
                kanaMaxTL = None
                kanaMaxBR = None

                # open each single kana image
                args = [ ( kana, kanaDirectory, kanasImage, kanasW, kanasH, method ) for kana in kanaFiles ]
                outputs = p.map( wrapper, args )

                # update estimated kana
                for output, kana in zip( outputs, kanaFiles ):
                    maxVal = output[0]
                    maxLoc = output[1]
                    kanaW = output[2]
                    kanaH = output[3]

                    if maxVal > kanaMaxVal:
                        kanaMaxVal = maxVal
                        kanaMaxName = kana
                        kanaMaxTL = maxLoc
                        kanaMaxBR = ( maxLoc[0] + kanaW, maxLoc[1] + kanaH )

                if kanaMaxVal < 0.95:
                    print( "%s   %s   %f" % ( kanasName, kanaMaxName, kanaMaxVal ) )
                    outlier += 1

                additionalDatas.append( kanaMaxTL[0] )
                additionalDatas.append( kanaMaxTL[1] )
                additionalDatas.append( kanaMaxBR[0] )
                additionalDatas.append( kanaMaxBR[1] )
                additionalDatas.append( kanaMaxVal )

            # write estimated bounding box and likelihood
            writer.writerow( [ imageName ] + annotations[i] + additionalDatas )

            nowtime = time.time()
            totalElapsed = nowtime - starttime
            sectionElapsed = nowtime - sectiontime
            sectiontime = nowtime
            print( "%dth Image Extension Finished [%d sec / total %d sec]" % ( i, sectionElapsed, totalElapsed ) )


    if outlier > 0:
        print( "outlier: %d" % outlier )

    elapsed = time.time() - starttime
    print( "Extension Finished [%d sec]" % elapsed )


if __name__ == "__main__":
    extend()
