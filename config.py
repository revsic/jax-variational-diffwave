from speechset.config import Config as DataConfig
from vlbdiffwave.config import Config as ModelConfig


class TrainConfig:
    """Configuration for training loop.
    """
    def __init__(self, sr: int, hop: int):
        """Initializer.
        Args:
            sr: sample rate, for relative segment size.
            hop: stft hop size.
        """
        # optimizer
        self.learning_rate = 2e-4
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.eps = 1e-9

        # 13000:100
        self.split = 13000
        self.bufsiz = 48

        # train iters
        self.epoch = 100

        # segment size
        seconds = 0.5
        self.segsize = int(sr * seconds) // hop * hop

        # path config
        self.log = './log'
        self.ckpt = './ckpt'

        # model name
        self.name = 'l1'

        # commit hash
        self.hash = 'unknown'


class Config:
    """Integrated configuration.
    """
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig(self.data.hop)
        self.train = TrainConfig(self.data.sr, self.data.hop)

    def dump(self):
        """Dump configurations into serializable dictionary.
        """
        return {k: vars(v) for k, v in vars(self).items()}

    @staticmethod
    def load(dump_):
        """Load dumped configurations into new configuration.
        """
        conf = Config()
        for k, v in dump_.items():
            if hasattr(conf, k):
                obj = getattr(conf, k)
                load_state(obj, v)
        return conf


def load_state(obj, dump_):
    """Load dictionary items to attributes.
    """
    for k, v in dump_.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj
