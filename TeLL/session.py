import os
import numpy as np
import tensorflow as tf
from TeLL.config import Config
from TeLL.utility.workingdir import Workspace
from TeLL.utility.timer import Timer
"""
© Michael Widrich, Markus Hofmarcher, 2017
"""

def set_seed(seed: int = 12345):
    tf.set_random_seed(seed)


class TeLLSession(object):
    def __init__(self, config: Config = None, summaries: list = ["training"], model_params=None):
        """
        Take care of initializing a TeLL environment.
            Creates working directory, instantiates network architecture, configures tensorflow and tensorboard.
            Furthermore takes care of resuming runs from an existing workspace.

        :param config: Config
            config object or None; if None config will be initialized from command line parameter
        :param summaries: list
            List of names for summary writers, by default one writer named "training" is opened
        :param model_params:
            Optional dictionary of parameters unpacked and passed to model upon initialization if not None

        :returns:

        tf_session: Tensorflow session

        tf_saver: Tensorflow checkpoint saver

        tf_summaries: dictionary containing tensorflow summary writers, accessible via the names passed upon creation

        model: TeLL model

        step: current global step (0 for new runs otherwise step stored in checkpoint file)

        workspace: TeLL workspace instance

        config: TeLL config object
        """
        if config is None:
            config = Config()
        
        # Setup working dir
        workspace = Workspace(config.working_dir, config.specs, config.restore)
        print("TeLL workspace: {}".format(workspace.working_dir))
        # Import configured architecture
        architecture = config.import_architecture()
        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get_value("cuda_gpu", "0"))
        # Some Tensorflow configuration
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=config.get_value("inter_op_parallelism_threads", 1),
            intra_op_parallelism_threads=config.get_value("intra_op_parallelism_threads", 1),
            log_device_placement=config.get_value("log_device_placement", False)
        )
        tf_config.gpu_options.allow_growth = config.get_value("tf_allow_growth", True)
        # Start Tensorflow session
        print("Starting session...")
        tf_session = tf.Session(config=tf_config)
        # Set tensorflow random seed
        set_seed(config.get_value("random_seed", 12345))
        #
        # Init Tensorboard
        #
        print("Initializing summaries...")
        summary_instances = {}
        for summary in summaries:
            summary_instances[summary] = tf.summary.FileWriter(os.path.join(workspace.get_tensorboard_dir(), summary),
                                                               tf_session.graph)
        # Initialize Model
        if model_params is None:
            model = architecture(config=config)
        else:
            model = architecture(config=config, **model_params)
        
        # Print number of trainable parameters
        trainable_vars = np.sum([np.prod(t.get_shape()) for t in tf.trainable_variables()])
        print("Number of trainable variables: {}".format(trainable_vars))
        
        with tf.name_scope("TeLL") as tell_namescope:
            # Store global step in checkpoint
            tf_global_step = tf.Variable(initial_value=tf.constant(0, dtype=tf.int64), name="tell_global_step",
                                         dtype=tf.int64, trainable=False)
            # Define placeholder and operation to dynamically update tf_global_step with a python integer
            global_step_placeholder = tf.placeholder_with_default(tf_global_step, shape=tf_global_step.get_shape())
            set_global_step = tf_global_step.assign(global_step_placeholder)
        
        #
        # Add ops to save and restore all the variables
        #
        tf_saver = tf.train.Saver(max_to_keep=config.get_value("max_checkpoints", 10), sharded=False)
        # Expose members
        self.tf_session = tf_session
        self.tf_saver = tf_saver
        self.tf_summaries = summary_instances
        self.model = model
        self.workspace = workspace
        self.config = config
        self.global_step = 0
        self.__global_step_placeholder = global_step_placeholder
        self.__global_step_update = set_global_step
        self.__tell_namescope = tell_namescope
    
    def initialize_tf_variables(self):
        """
        Initialize tensorflow variables (either initializes them from scratch or restores from checkpoint).
        
        :return: updated TeLL session
        """
        session = self.tf_session
        checkpoint = self.workspace.get_checkpoint()
        #
        # Initialize or load variables
        #
        with Timer(name="Initializing variables"):
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())
        
        if checkpoint is not None:
            # restore from checkpoint
            self.tf_saver.restore(session, checkpoint)
            # get step number from checkpoint
            step = session.run(self.__global_step_placeholder) + 1
            self.global_step = step
            # reopen summaries
            for _, summary in self.tf_summaries.items():
                summary.reopen()
                summary.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=step)
            print("Resuming from checkpoint '{}' at iteration {}".format(checkpoint, step))
        else:
            for _, summary in self.tf_summaries.items():
                summary.add_graph(session.graph)
        
        return self
    
    def save_checkpoint(self, global_step: int):
        """
        Store current state in checkpoint.
        
        :param global_step: current global step variable; has to be stored for resuming later on.
        """
        self.global_step = global_step
        tf_session = self.tf_session
        # Update global step variable
        tf_session.run(self.__global_step_update, feed_dict={self.__global_step_placeholder: global_step})
        # Store checkpoint
        self.tf_saver.save(tf_session,
                           os.path.join(self.workspace.get_result_dir(), "checkpoint-{}.ckpt".format(global_step)))
    
    def close(self, save_checkpoint: bool = True, global_step: int = 0):
        """
        Close all tensorflow related stuff and store checkpoint if requested.
        
        :param save_checkpoint: bool
            flag indicating whether current state should be stored in checkpoint
        :param global_step: int
            if save_checkpoint is True this value is required to store the step in the checkpoint
        """
        tf_session = self.tf_session
        # Close Summarywriters
        for _, summary in self.tf_summaries.items():
            summary.close()
        # Save checkpoint
        if save_checkpoint:
            self.save_checkpoint(global_step)
        # Close session
        tf_session.close()
