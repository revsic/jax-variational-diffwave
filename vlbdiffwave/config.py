class Config:
    """Configuration for VLB-DiffWave implementation.
    """
    def __init__(self):
        # embedding config
        self.embedding_size = 128
        self.embedding_proj = 512
        self.embedding_layers = 2
        self.embedding_factor = 128

        # upsampler config
        self.upsample_strides = [16, 1]
        self.upsample_kernels = [32, 3]
        self.upsample_layers = 2
        # computed hop size
        self.hop = self.upsample_stride[0] ** self.upsample_layers

        # block config
        self.channels = 64
        self.kernels = 3
        self.dilations = 2
        self.num_layers = 10
        self.num_cycles = 3
