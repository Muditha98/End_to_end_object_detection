
import zipfile

with zipfile.ZipFile('signlang_dataset_labelled.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

import shutil

shutil.move("signlang_dataset_labelled/train", "./")