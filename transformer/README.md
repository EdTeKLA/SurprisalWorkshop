# Transformer

Most of the code contained here is based on the work of Merkx & Frank (2021):
> https://github.com/DannyMerkx/next_word_prediction

> [Human Sentence Processing: Recurrence or Attention?](https://aclanthology.org/2021.cmcl-1.2) (Merkx & Frank, CMCL 2021)



I made minor changes to the code, such as:
* Updating file paths
* Changing the Transformer configuration
* Increasing the reporting of training and validation loss
* Enclosing code in a for loop for hyperparameter tuning

The following files are my own work: 
* [format_surprisal_output.py](format_surprisal_output.py)
* [hyperparameter_selection.py](hyperparameter_selection.py)
* [calculate_test_perplexity.py](caclulate_test_perplexity.py)



