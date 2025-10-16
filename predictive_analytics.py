import random

import hashlib

import pandas as pd

from sklearn.model_selection import

train_test_split

import optuna# -------------------------------

# Generate 10 Random Seeds

# -------------------------------

def md5_hash(input_string):

"""Generate MD5 hash from a string."""

md5_hasher = hashlib.md5()

md5_hasher.update(input_string.encode('utf-8'))

return md5_hasher.hexdigest()

# Use your name as seed generator

input_string = "Mamta"

hashed_value = md5_hash(input_string)

number = int(hashed_value, 16)random.seed(number)

seeds = [random.randint(0, 2**31 - 1) for _ in

range(10)]

print("Generated Seeds:", seeds)

# -------------------------------

# Load Dataset

# -------------------------------

# Example: MIMIC readmission dataset

(preprocessed with `readmission` column)

df = pd.read_csv("mimic_readmission.csv")

X = df.drop(columns=["readmission"])

y = df["readmission"]

# #  Train/Valid/Test Split (60:20:20) for each

seed

# -------------------------------

splits = {}

for i, seed in enumerate(seeds, start=1):

# Train (60%) + Temp (40%)

X_train, X_temp, y_train, y_temp =

train_test_split(

X, y, test_size=0.4, random_state=seed,

stratify=y

)

#Validation (20%) + Test (20%)

X_valid, X_test, y_valid, y_test =

train_test_split(X_temp, y_temp, test_size=0.5,

random_state=seed, stratify=y_temp

)

splits[f"Seed_{i}"] = {

"train": (X_train, y_train),

"valid": (X_valid, y_valid),

"test": (X_test, y_test)

}

print(f"\nSplit with Seed {seed}:")

print(f"Train: {X_train.shape}, Valid:

{X_valid.shape}, Test: {X_test.shape}")

# -------------------------------

# Example: Access Seed 1 Split# -------------------------------

X_train, y_train = splits["Seed_1"]["train"]

X_valid, y_valid = splits["Seed_1"]["valid"]

X_test, y_test = splits["Seed_1"]["test"]

print("\nReady for Optuna hyperparameter

tuning!")