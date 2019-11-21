
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import time
import resnet
import utils


def train():
    # load data: dataset/train_kana/ includes 388,146 images
    trainFolder = "D:/ImageLab/Camp/2019/alcon2019/dataset/train_kana/"
    train_batch_size = 8

    transform = transforms.Compose( [ transforms.Resize( ( 224, 224 ) ), transforms.ToTensor(), transforms.Normalize( [ 0.5 ], [ 0.25 ] ) ] )
    # transform = transforms.Compose( [ transforms.Resize( ( 224, 224 ) ), transforms.ToTensor(), transforms.Normalize( ( 0.485, 0.456, 0.406 ), ( 0.229, 0.224, 0.225 ) ) ] )
    trainset = resnet.KanaDataset( trainFolder, transform )
    trainloader = torch.utils.data.DataLoader( trainset, batch_size=train_batch_size, shuffle=True, num_workers=2 )

    classes = trainset.getClasses()
    classNum = len( classes )
    # for i, c in enumerate( classes, 0 ):
    #     print( "%d %s %s" % ( i, c, utils.unicode2kana( c ) ) )

    # use CUDA
    device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
    print( device )

    # generate net
    net = resnet.Net( outputFeatures=classNum )
    net = net.to( device )

    # set parameters
    criterion = nn.CrossEntropyLoss() # nn.MSELoss()
    optimizer = optim.SGD( net.parameters(), lr=0.1, momentum=0.9 )

    # load parameters
    net.load_state_dict( torch.load( "result/122_weight.pth" ) )
    optimizer.load_state_dict( torch.load( "result/122_optimizer.pth" ) )

    # train
    starttime = time.time()
    print( "Start Training" )

    epochNum = 128
    for epoch in range( 122, epochNum ):
        runningloss = 0.0
        for i, data in enumerate( trainloader, 0 ):
            inputs, labels = data[ 0 ].to( device ), data[ 1 ].to( device )

            optimizer.zero_grad()
            outputs = net( inputs )

            loss = criterion( outputs, labels )
            loss.backward()
            optimizer.step()

            runningloss += loss.item()
            if i % 2000 == 1999:
                print( "[%d, %5d] loss: %.3f" % ( epoch + 1, i + 1, runningloss / 2000 ) )
                runningloss = 0.0

        torch.save( net.state_dict(), "result/%s_weight.pth" % str( epoch + 1 ).zfill( len( str( epochNum ) ) ) )
        torch.save( optimizer.state_dict(), "result/%s_optimizer.pth" % str( epoch + 1 ).zfill( len( str( epochNum ) ) ) )

    elapsed = time.time() - starttime
    print( "Training Finished [%d sec]" % elapsed )

    # save trained-model
    torch.save( net.state_dict(), "result/weight.pth" )
    torch.save( optimizer.state_dict(), "result/optimizer.pth" )

    return


def test():
    # load data
    testFolder = "D:/ImageLab/Camp/2019/alcon2019/dataset/train_kana/"
    test_batch_size = 8

    transform = transforms.Compose( [ transforms.Resize( ( 224, 224 ) ), transforms.ToTensor(), transforms.Normalize( [ 0.5 ], [ 0.25 ] ) ] )
    # transform = transforms.Compose( [ transforms.Resize( ( 224, 224 ) ), transforms.ToTensor(), transforms.Normalize( ( 0.485, 0.456, 0.406 ), ( 0.229, 0.224, 0.225 ) ) ] )
    testset = resnet.KanaDataset( testFolder, transform, isTest=True )
    testloader = torch.utils.data.DataLoader( testset, batch_size=test_batch_size, shuffle=False, num_workers=2 )

    classes = testset.getClasses()
    classNum = len( classes )
    # for i, c in enumerate( classes, 0 ):
    #     print( "%d %s %s" % ( i, c, utils.unicode2kana( c ) ) )

    # use CUDA
    device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
    print( device )

    # generate net
    net = resnet.Net( outputFeatures=classNum )
    net.load_state_dict( torch.load( "result/48_weight.pth" ) )
    net = net.to( device )

    # test using by train dataset
    starttime = time.time()
    print( "Start Testing" )

    all_correct = 0
    all_total = 0
    class_correct = list( 0. for i in range( classNum ) )
    class_total = list( 0. for i in range( classNum ) )
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to( device )
            labels = labels.to( device )
            outputs = net( images )
            _, predicted = torch.max( outputs, 1 )
            corrects = ( predicted == labels ) #.squeeze()

            all_correct += corrects.sum().item()
            all_total += labels.size( 0 )

            for i, label in enumerate( labels, 0 ):
                class_correct[ label ] += corrects[ i ].item()
                class_total[ label ] += 1

    elapsed = time.time() - starttime
    print( "Testing Finished [%d sec]" % elapsed )
    print( "Accuracy of the network on the %d test images: %d %% ( %d / %d )" % ( len( testset ), 100 * all_correct / all_total, all_correct, all_total ) )

    for i in range( classNum ):
        print( "Accuracy of %5s(%s) : %2d %% ( %d / %d )" % ( classes[ i ], utils.unicode2kana( classes[ i ] ), 100 * class_correct[ i ] / class_total[ i ], class_correct[ i ], class_total[ i ] ) )

    return


if __name__ == "__main__":
    train()
    test()
