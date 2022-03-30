from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import yaml
import numpy as np
import math
from enum import Enum
import heapq


class FrameData:
    pass


class Container:
    pass


class ConfigParser(object):
    def __init__(self, config):
        stream = open(config,'r')
        docs = yaml.load_all(stream)
        for doc in docs:
            for k, v in doc.items():
                cmd = "self." + k + "=" + 'Container()'
                exec(cmd)
                for k1, v1 in v.items():
                    cmd = "self." + k + '.' + k1 + "=" + repr(v1)
                    print(cmd)
                    exec(cmd)
        stream.close()


def create_frame_data(frame_width, frame_height):
    frame_data = FrameData()
    frame_data.frame_width = frame_width
    frame_data.frame_height = frame_height
    frame_data.cx = frame_data.frame_width / 2.0
    frame_data.cy = frame_data.frame_height / 2.0
    frame_data.fx = 500.0 * (frame_data.frame_width / 640.0)
    frame_data.fy = 500.0 * (frame_data.frame_height / 480.0)
    frame_data.fx = (frame_data.fx + frame_data.fy) / 2.0
    frame_data.fy = frame_data.fx

    return frame_data


def euler_angles_to_rotation_matrix(theta):
    r_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    r_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    r_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    r = np.dot(r_z, np.dot(r_y, r_x))

    return r


def extract_mouth_from_frame(frame, mouth_anchor, mouth_height, mouth_width):
    mouth_top = np.rint(mouth_anchor[1]).astype(np.int32)
    mouth_bottom = mouth_top + mouth_height
    mouth_left = np.rint(mouth_anchor[0] - int(mouth_width / 2)).astype(np.int32)
    mouth_right = mouth_left + mouth_width

    mouth = frame[mouth_top:mouth_bottom, mouth_left:mouth_right]

    assert mouth.shape == (mouth_height, mouth_width, 3)

    return mouth


class DBType(Enum):
    Train = 0,
    Validation = 1,
    Test = 2,
    Example = 3


class Sequence(object):
    """Represents a complete or partial sequence."""

    def __init__(self, output=None, decoder_output=None, decoder_hidden=None, decoder_cell=None, attn_context=None, encoded_sequence=None, logprob=None, score=None):
        """Initializes the Sequence.

        Args:
          output: List of word ids in the sequence.
          state: Model state after generating the previous word.
          logprob: Log-probability of the sequence.
          score: Score of the sequence.
        """
        self.output = output
        self.decoder_output = decoder_output
        self.decoder_hidden = decoder_hidden
        self.decoder_cell = decoder_cell
        self.attn_context = attn_context
        self.encoded_sequence = encoded_sequence
        self.logprob = logprob
        self.score = score

    def __cmp__(self, other):
        """Compares Sequences by score."""
        assert isinstance(other, Sequence)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Sequence)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Sequence)
        return self.score == other.score


class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.

        The only method that can be called immediately after extract() is reset().

        Args:
          sort: Whether to return the elements in descending sorted order.

        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []