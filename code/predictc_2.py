import pandas as pd
import base64

def get_top_classes_by_confidence(df, top_x):
    """
    Returns a dictionary of the top `top_x` classes by confidence from a DataFrame.

    Parameters:
    - df (pd.DataFrame): A DataFrame with 'class' (str) and 'confidence' (float) columns.
    - top_x (int): The number of top classes to return.

    Returns:
    - dict: A dictionary with class names as keys and confidence as values, sorted by confidence in descending order.
    """
    if 'class' not in df.columns or 'confidence' not in df.columns:
        raise ValueError("DataFrame must have 'class' and 'confidence' columns.")
    
    top_classes = df.nlargest(top_x, 'confidence')
    return dict(zip(top_classes['class'], top_classes['confidence']))

def get_prediction_data(learner, img_path, output):
  # predict classes for the image, suppressing progress output
  with learner.no_bar():
    x = learner.predict(img_path)

  # put results into a dataframe to do better manipulation

  # confidence numbers from tensors into a list
  conf = [tensor.item() for tensor in x[2]]
  # make the dataframe with the classes and confidences
  df = pd.DataFrame(zip(learner.dls.vocab, conf), columns=['class', 'confidence'])
  # filter for 2-digit classes
  two_digit_classes = df[df['class'].str.match(r'^\d{2}$')]
  # filter for 1-digit classes
  one_digit_classes = df[df['class'].str.match(r'^\d$')]
  # Find the class with the highest confidence and return just the string
  highest_conf_class = two_digit_classes.loc[two_digit_classes['confidence'].idxmax(), 'class']

  # get some of the top classes to evaluate

  top_x = 5
  top_all = get_top_classes_by_confidence(df, top_x)

  top_x = 3
  top_2_digit = get_top_classes_by_confidence(two_digit_classes, top_x)

  top_x = 3
  top_1_digit = get_top_classes_by_confidence(one_digit_classes, top_x)

  pred_results = {
      "pred": highest_conf_class,
      "top_all": top_all,
      "top_2_digit": top_2_digit,
      "top_1_digit": top_1_digit
  }

  output["custom_model"] = highest_conf_class

  output["top_all_classes"] = top_all
  output["top_2_digit_classes"] = top_2_digit
  output["top_1_digit_classes"] = top_1_digit 

  return pred_results
