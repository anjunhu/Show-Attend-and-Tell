# Show, Attend and Tell: Neural Image Caption Generation with Visual Attention

#### Pre-trained Decoders

- [VGG19](https://www.dropbox.com/s/eybo7wvsfrvfgx3/model_10.pth?dl=0)
- [ResNet152](https://www.dropbox.com/s/0fptqsw3ym9fx2w/model_resnet152_10.pth?dl=0)
- [ResNet152 No Teacher Forcing](https://www.dropbox.com/s/wq0g2oo6eautv2s/model_nt_resnet152_10.pth?dl=0)
- [VGG19 No Gating Scalar](https://www.dropbox.com/s/li4390nmqihv4rz/model_no_b_vgg19_5.pth?dl=0)

#### COCO Dataset and Annotations  

[COCO2014 Captioning data](https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip)   
Decompress and place `dataset.json` in `data/coco`

#### Usage

Run the preprocessing to create the needed JSON files:

```bash
python generate_json_data_iot_from_coco.py  # create json file using coco dictionary.

python generate_json_data_iot_from_scratch.py  # create json file using iot vocabulary only. Approximately 240 words (>5 occurrences) or 600 words (all unique words) .

```

Start the training or fine-tuning from an existing checkpoint:

```bash
python train.py --network resnet152 --data ./data/cocodict-iot --lr 1e-4 # training from scratch

python train.py --network resnet152 --model model_coco/model_resnet152_10.pth --data ./data/cocodict-iot --lr 5e-6 # fine-tuning

```

The models will be saved in `model/` and the training statistics will be saved in `runs/`.    

To run inference on an IoT dataset: 

```bash
python generate_caption_multi.py --network resnet152 --model model/model_resnet152_10.pth --dict-path data/coco/word_dict_coco.json --visualize # pop up window visualization

python generate_caption_multi.py --network resnet152 --model model/model_resnet152_10.pth --dict-path data/coco/word_dict_coco.json > model_resnet152_10_prediction_IMERIT_2024_03_25_TEST_v5.csv # saving results for metric calculation

```
Then use `data/iot/metrics_calculation.ipynb` to calculate BLEU, ROUGE and CIDEr metrics.


## References

[Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf)

[Original Theano Implementation](https://github.com/kelvinxu/arctic-captions)

[Original PyTorch Implementation](https://github.com/AaronCCWong/Show-Attend-and-Tell)

[Neural Machine Translation By Jointly Learning to Align And Translate](https://arxiv.org/pdf/1409.0473.pdf)

[Karpathy's Data splits](https://cs.stanford.edu/people/karpathy/deepimagesent/)
