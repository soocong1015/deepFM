import matplotlib.pyplot as plt

def test_plot(test_output, test_pred_output):
    plt.plot(test_output.reset_index(drop=True))
    plt.plot(test_pred_output)
    plt.show()

def loss_plot(histo):
    plt.plot(histo.history['loss'])
    plt.plot(histo.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()