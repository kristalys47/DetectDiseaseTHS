import numpy as np
def main():
    ksadno = [[1,2],[3,4]]
    asdf = np.array(ksadno)
    print(asdf.shape)

    GHJK= []

    GHJK.append(ksadno)
    GHJK.append(ksadno)
    GHJK.append(ksadno)

    kbk = np.array(GHJK)

    np.save("array", kbk)
    print(kbk)
    print(kbk.shape)
    hello = np.load("array.npy")
    print(hello)


#joderme
if __name__ == "__main__":
    main()