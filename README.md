# Multi_Label_Text_Classification

## Dependencies

python==3.8.10

pytorch==2.4.1

transformer==4.46.2

numpy==1.24.2

## Training and testing

1. Download this repository to local.

2. Download **pytorch_model.bin** from [https://huggingface.co/google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/main), and add it to **code/bert-base-uncased** folder.

3. Add **code/fintune_models** folder.

4. Add multi-label text classification dataset to **data** folder. 

5. Run the **train.py** for training.

6. Run the **train.py** for testing.

## Tips

* **bert-base-uncased**: [google-bert/bert-base-uncased · Hugging Face.](https://huggingface.co/google-bert/bert-base-uncased)

* You can change model structure by modifying **model.py**.

* There may be some warnings during the code execution, which you can temporarily ignore.

* If there is any problem with the code, you can contact me through 1500519790@qq.com.

