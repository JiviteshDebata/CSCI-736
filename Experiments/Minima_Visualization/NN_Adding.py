
import torch 

class NN:
    def __init__(self) -> None:
        pass

    def test(self):
        if torch.cuda.is_available():
            print('We have a GPU!')
        else:
            print('Sorry, CPU only.')


def main():
    net = NN()
    net.test()


if __name__ == "__main__":
    main()