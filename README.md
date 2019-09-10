This repository contains code for [my blog post](https://towardsdatascience.com/hello-world-in-speech-recognition-b2f43b6c5871?source=friends_link&sk=39c0b0d8d3655428f8273e7d2d466eca)

Note: This code uses tensorflow 2.0

Files present:  
`bare_bones_asr.py`:  
* code for the neural network explained in the blog  
* code for loading `sample.wav` file and creating its spectrogram  
* code for training given neural network with `sample.wav` and its transcript as input using CTC loss  
* code for using trained neural network to predict on an input spectrogram  
  
`prefix_beam_search.py`:  
* code for prefix beam search as explained in the blog  
* you can import the function from this file directly and use it on your ctc output  
```
from prefix_beam_search import prefix_beam_search
example_ctc_output = None  # get your ctc output from the network
alphabet = list(ascii_lowercase) + [space_token, end_token, blank_token]  # get your character vocab
lm = None  # get your language model function
print(prefix_beam_search(example_ctc, alphabet, blank_token, end_token, space_token, lm=lm))
```
