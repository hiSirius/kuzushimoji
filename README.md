# kuzushimoji
## くずし字認識チャレンジ2019 

https://sites.google.com/view/alcon2019/ホーム?authuser=0

### 三文字のくずし字認識：

K-meanで文字を分割して、一文字のデータセットから学習したモデルを用いて各文字を認識する
- K-meansの分割結果


![分割の結果](https://github.com/hiSirius/kuzushimoji/blob/master/data/split_test/10._test.jpg)

### 三文字のくずし文字認識精度：約40%

### データセットの取得
```
wget http://www.iic.ecei.tohoku.ac.jp/~tomo/alcon2019.tar.gz
tar xzvf alcon2019.tar.gz
```
