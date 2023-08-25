import matplotlib.pyplot as plt
import numpy as np


loss = np.load('result/loss.npy')
acc = np.load('result/acc.npy')

x_axis = np.arange(len(loss)) + 1

fig1 = plt.figure()
plt.plot(x_axis, loss, linewidth=2, marker='o', markersize=10, color='orange')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Training Loss', fontsize=15)
# plt.legend()
fig1.suptitle('Training Loss', fontsize=20)
fig1.savefig('loss.png')

fig2 = plt.figure()
plt.plot(x_axis, acc, linewidth=2, marker='o', markersize=10, color='blue')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Test Accuracy', fontsize=15)
# plt.legend()
fig2.suptitle('Test Accuracy', fontsize=20)
fig2.savefig('acc.png')
plt.show()