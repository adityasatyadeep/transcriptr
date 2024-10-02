import os
import modal

app = modal.App("hello world")

@app.function()
def square(x):
    return x**2

@app.local_entrypoint()
def main():
    x = 6
    x_squared = square.remote(x)
    print(x_squared) 


# MODEL_DIR = os.path.join("model")
# print(MODEL_DIR)