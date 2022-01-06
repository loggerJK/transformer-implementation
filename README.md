# transformer-implementation
Vanilla Transformer NLP 모델을 구현합니다.

[Paper : Attention is All You Need](https://arxiv.org/abs/1706.03762)

- Framework : Pytorch
- Task : EN -> KO Translation
- Dataset : OpenSubtitle Corpus (in Korpora)
    


# To Do
- [Done] device configuration 
- [Done] Training / Loss Function
= [Done] Mask Functions
- Positional Embedding Layer
- [Done] Dropout
- [Done] Dataset 제대로 동작하는지 확인
- LayerNorm hidden_dim 의미


# About Dataset

## OpenSubtitle Dataset Description
Author : TRAC (https://trac.edgewall.org/)

Repository : http://opus.nlpl.eu/OpenSubtitles-v2018.php
References :
    - P. Lison and J. Tiedemann, 2016, OpenSubtitles2016: Extracting Large Parallel Corpora
        from Movie and TV Subtitles. In Proceedings of the 10th International Conference on
        Language Resources and Evaluation (LREC 2016)

This is a new collection of translated movie subtitles from http://www.opensubtitles.org/.

[[ IMPORTANT ]]


If you use the OpenSubtitle corpus: Please, add a link to http://www.opensubtitles.org/
to your website and to your reports and publications produced with the data!

I promised this when I got the data from the providers of that website!

This is a slightly cleaner version of the subtitle collection using improved sentence alignment
and better language checking.


## License
Open Data. Details in https://opendefinition.org/od/2.1/en/