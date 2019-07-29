from matplotlib import pyplot as plt

def show_image(img):
    fig = plt.figure(figsize=(8,3))
    fig.add_subplot(1, 1, 1)
    plt.imshow(img)
    plt.show()
