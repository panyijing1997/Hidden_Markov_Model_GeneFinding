from training_by_counting import load_model
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf) # 加上这一句
model = load_model('models/validated_on_3')
print("init probs")
print(model.init_probs)
print("trans probs")
print(model.trans_probs)
print("emission probs")
print(model.emission_probs)

with open('model_printed.txt' , 'w') as f:
    f.write(str(model.init_probs))
    f.write(str(model.trans_probs))
    f.write(str(model.emission_probs))
