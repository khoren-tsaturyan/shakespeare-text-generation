# Text Generation in Shakespeare Style
This is the code for generating Shakespeare style text. 

For this I fine-tuned GPT2 model with pytorch using dataset from kaggle(https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays).

Fine-tuned model with tokanization files available in results folder. Model was trained for 3 epochs with batch size 2 and learning rate 0.0005.
To see more about the training process, check out the log files in the logs directory.

I also deployed model using streamlit app, so you can see how it performs by follwing this link.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([URL_TO_YOUR_APP](https://shakespeare-text-generation.streamlit.app/))
## Training
For fine-tuning the gpt2 model you can run main.py file.
```python main.py```

Also you can modify following parameters.

```--batch-size``` - input batch size for training (default: 2)

```--epochs``` - number of epochs to train (default: 3)

```--lr``` - learning rate step size (default: 5e-4)

```--seed``` - random seed (default: 42)

```--no-cuda``` - disables CUDA training

```--save-mode``` - for saving the current model

#### Example

```python main.py --batch-size=8 --epochs=5 --save-model```

## Generate text
You can also genrate text by running generate.py file.

#### Example

```python generate text.py --output-length=700 --temperature=1```
