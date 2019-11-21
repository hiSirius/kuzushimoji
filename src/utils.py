
def unicode2kana( code:str ):
    assert len( code ) == 6
    return chr( int( code[2:], 16 ) )


def kana2unicode( kana:str ):
    assert len( kana ) == 1
    return 'U+' + hex( ord( kana ) )[2:]


def unicodes2kanas( codes:list ):
    assert len( codes ) == 3
    return [ unicode2kana( x ) for x in codes ]


def evaluation( y0, y1 ):
    cols = [ "Unicode1", "Unicode2", "Unicode3" ]
    x = y0[ cols ] == y1[ cols ]
    x2 = x[ "Unicode1" ] & x[ "Unicode2" ] & x[ "Unicode3" ]
    acc = sum( x2 ) / len( x2 ) * 100
    return acc
