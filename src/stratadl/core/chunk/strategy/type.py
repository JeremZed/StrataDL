from enum import Enum

class ChunkTypeStrategy(Enum):
    """ Représente les différentes stratégie de chunking"""
    CLASSIC = "classic"
    SENTENCE = "sentence"
    TOKEN = "token"
    HIERARCHY = "hierarchy"