from torch.autograd import Variable
from torch import Tensor



def position_encoding(embedding_size, sentence_size):
    # dt = CudaTensor if cuda else Tensor
    encoding = Tensor(embedding_size, sentence_size)
    encoding = Variable(encoding)

    for i, j in [(i, j) for
                 i in range(embedding_size) for
                 j in range(sentence_size)]:
        encoding[i, j] = (
            ((i+1) - (embedding_size+1)/2) *
            ((j+1) - (sentence_size+1)/2)
        )
    encoding *= 4 / (embedding_size * sentence_size)
    encoding[:, -1] = 1.
    return encoding.t()

def temporal_encoding(memory_size, embedding):
    time = Tensor(range(memory_size))
    time = Variable(time)
    return embedding(time)
