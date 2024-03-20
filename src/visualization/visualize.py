import matplotlib.pyplot as plt


def plot_losses(epochs:int, losses:list[float], name:str)->None:
    plt.figure(figsize=(8,8))
    plt.plot(range(epochs), losses)
    plt.title("Losses")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.grid(True)

    plt.savefig("./results/" + name + "_losses.png")