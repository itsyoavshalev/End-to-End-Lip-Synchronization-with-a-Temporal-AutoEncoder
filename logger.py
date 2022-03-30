import tensorflow as tf
import os


class Logger:
    def __init__(self, config):
        log_dir = os.path.join(config.general.output_path, config.general.logs_folder)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.writer = tf.summary.FileWriter(log_dir)

    def dump_current_errors(self, errors, step):
        for tag, value in errors.items():
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.writer.add_summary(summary, step)
