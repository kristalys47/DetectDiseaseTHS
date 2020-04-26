from ths.utils.files import GloveEmbedding
from ths.utils.sentences import SentenceToIndices, SentenceToEmbedding, PadSentences

def main():
    G = GloveEmbedding("../test/data/glove.6B.50d.txt")
    word_to_idx, idx_to_word, embedding = G.read_embedding()
    #print("locon: ", word_to_idx["locon"])
    print("Length dictionary: ", len(word_to_idx))
    s = "I love New York and music locon"
    s = s.lower()
    print("Sentence: ", s)
    S = SentenceToIndices(word_to_idx)

    e = "I love New York and music locon"
    e = e.lower()
    print("Sentence: ", e)
    E = SentenceToIndices(word_to_idx)
    matrices = []

    sentence = S.map_sentence(s)
    print("Sentence to indices: ", sentence)
    print("Padded: ", PadSentences(10).pad(sentence))
    SE = SentenceToEmbedding(word_to_idx, idx_to_word, embedding)
    matrix = SE.map_sentence(s, max_len=10)
    matrixe = SE.map_sentence(e, max_len=10)
    matrices.append(matrix)
    matrices.append(matrixe)

    print("Matrix: ", matrix)
    print("Matrix.shape: ", matrix.shape)
    print("Embedding i: ", embedding[word_to_idx["i"]])

    sentences = []
    sentences.append("I esta malo".lower())
    sentences.append("Love la musica salsa.".lower())
    sentences.append("Uff, q mal te va nene".lower())
    mapped, mlen = S.map_sentence_list(sentences)
    print("mlen: ", mlen)
    for s in mapped:
        print(s)

    print("Embedding 0: ", embedding[0])
    print("Embedding 400001: ", embedding[400001])

if __name__ == "__main__":
    main()