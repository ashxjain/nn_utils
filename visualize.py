from matplotlib import pyplot as plt

def show_image(img, figsize=(2, 2), annotation=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(img)
    plt.imshow(img)
    if annotation:
        plt.annotate(annotation, xy=(0,0), xytext=(0,-1.2), fontsize=13)

def show_class_image(class_labels, images, labels, filter_class, count=5, figsize=(2, 2)):
    cnt = 0
    for idx in range(len(labels)):
        if class_labels[labels[idx]] == filter_class:
            show_image(images[idx])
            cnt += 1
        if cnt == count:
            break

def show_misclassified_images(class_labels, actual, predicted, images, max_cnt=5):
    cnt = 0
    for idx in range(len(predicted)):
        i_pred, i_act = predicted[idx], actual[idx]
        if i_pred != i_act:
            annotation = "Actual: %s, Predicted: %s" % (class_labels[i_act], class_labels[i_pred])
            show_image(images[idx], annotation=annotation)
            cnt += 1
        if cnt == max_cnt:
            return

def show_misclassified_classes(class_labels, actual, predicted):
    misclass = {}
    for idx in range(len(predicted)):
        i_pred, i_act = predicted[idx], actual[idx]
        if i_pred != i_act:
            cl = class_labels[i_act]
            misclass[cl] = misclass.get(cl, 0) + 1
    for k, v in misclass.items():
        print(k, v)

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

def current_accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)
