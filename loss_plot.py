import json

name = 'ResUNet'
# name = 'UNet'

history_file = open(name + '.json', "r")
history = history_file.read()
history_file.close()

history = json.loads(history)

import matplotlib.pyplot as plt
plt.plot(history['train'])
plt.plot(history['valid'])
plt.legend(['train loss', 'valid loss'])
plt.show()