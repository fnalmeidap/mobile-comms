from dataclasses import dataclass

@dataclass
class ModelNN:
    L1_INPUT_SIZE = 28*28
    L1_OUTPUT_SIZE = 512
    L2_INPUT_SIZE = L1_OUTPUT_SIZE
    L2_OUTPUT_SIZE = 512
    L3_INPUT_SIZE = L2_OUTPUT_SIZE
    L3_OUTPUT_SIZE = 10

@dataclass
class Optimizer:
    LEARNING_RATE = 1e-3