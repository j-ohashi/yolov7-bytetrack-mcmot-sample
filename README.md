# yolov7-bytetrack-mcmot-sample

Kazuhito00さんの[yolox-bytetrack-mcmot-sample](https://github.com/Kazuhito00/yolox-bytetrack-mcmot-sample)の物体検出部分をYOLOXからibaiGorordoさんの[ONNX-YOLOv7-Object-Detection](https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection)に入れ替えたものです。

必要モジュール等は上記2リポジトリに準じます。  

## 実行方法
ONNX-YOLOv7-Object-DetectionからYOLOv7ディレクトリをコピーし、学習モデルをmodelディレクトリに格納した上で、yolov7_sample.pyを実行します。  
学習モデルは、[PINTO0309/PINTO_model_zoo/307_YOLOv7](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/307_YOLOv7)の`download_single_batch.sh`を実行して得られるonnxファイルを使用します。  

- The License of the models is GPL-3.0 license: [License](https://github.com/WongKinYiu/yolov7/blob/main/LICENSE.md)
```
python yolov7_sample.py
```

## 実行時パラメータ
YOLOv7パラメータ  
- --model  
ロードするモデルの格納パス  
デフォルト：model/yolov7_384x640.onnx
- --score_th  
クラス判別の閾値  
デフォルト：0.3  
- --nms_th  
NMSの閾値  
デフォルト：0.45

その他の入力関連パラメータ、ByteTrackパラメータはyolox-bytetrack-mcmot-sampleに準じます。

## Reference
- [Kazuhito00/yolox-bytetrack-mcmot-sample](https://github.com/Kazuhito00/yolox-bytetrack-mcmot-sample)
- [ibaiGorordo/ONNX-YOLOv7-Object-Detection](https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection)
- [PINTO0309/PINTO_model_zoo/307_YOLOv7](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/307_YOLOv7)

## Author
Jun Ohashi ([Twitter](https://twitter.com/Jun_Ohashi) / [Facebook](https://www.facebook.com/jun.ohashi))

## License
yolov7-bytetrack-mcmot-sample is under MIT License