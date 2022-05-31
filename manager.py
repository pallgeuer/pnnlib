# PNN Library: Neural network manager

# Imports
import os
import gc
import re
import io
import csv
# noinspection PyCompatibility
import pwd
import sys
import math
import json
import yaml
import stat
import pickle
import base64
import hashlib
import os.path
import inspect
import argparse
import datetime
import platform
import itertools
import contextlib
import statistics
import subprocess
import collections
import dataclasses
from enum import Enum, auto
from typing import Any, Union, Tuple, Iterable, Callable, Dict
import portalocker
import torch
import torch.backends.cudnn
import ppyutil.git
import ppyutil.nvsmi
import ppyutil.objmanip
import ppyutil.interpreter
import ppyutil.contextman
import ppyutil.execlock
import ppyutil.filter
import ppyutil.pickle
import ppyutil.print
import ppyutil.string
import ppyutil.stdtee
from ppyutil.classes import EnumLU
from ppyutil.string import strtobool
from ppyutil.print import print_warn
from ppyutil.argparse import AppendData, IntRange
from pnnlib import config, yaml_spec, netmodel, dataset, training, GPUHardwareError
from pnnlib.training import loss_fmt, rloss_fmt
from pnnlib.util import system_info, device_util, tensor_util, model_util, misc_util
from pnnlib.util.device_util import GPULevel

# Conditionally import tensorwatch
try:
	import tensorwatch
except ImportError:
	tensorwatch = None

####################
### Enumerations ###
####################

# Context enumeration
class Context(Enum):
	Process = auto()
	Action = auto()

# Actions enumeration
class Action(Enum):
	DebugArgs = auto()
	LoadModel = auto()
	KeepModel = auto()
	ResetModel = auto()
	ShowConfigs = auto()
	ShowCSVFmts = auto()
	GitSnapshot = auto()
	ModelInfo = auto()
	DrawModel = auto()
	DatasetInfo = auto()
	DatasetStats = auto()
	Train = auto()
	TrainAll = auto()
	PerfModel = auto()
	PerfModelOptim = auto()

# Configuration enumeration
class Configuration(Enum):
	Default = auto()
	NonDefault = auto()
	All = auto()

# Learning rate scheduler arguments enumeration
class LRSchedulerArgs(Enum):
	NoArgs = auto()
	ValidLoss = auto()
	MinRefValidLoss = auto()

# Dataset statistics basis
class DatasetStatsBasis(EnumLU):
	Element = auto()
	Sample = auto()
	Batch = auto()
	Default = Sample

#################
### Constants ###
#################

# Config parameter specifications
pnn_config_spec = {
	'CudnnBenchmarkMode': bool,       # Whether to use cuDNN benchmark mode (https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936, ignored and always false if deterministic)
	'Deterministic': bool,            # Whether all random number generation should be performed deterministically
	'DeterministicSeed': int,         # Seed to use for deterministic random number generation
	'GPUTempLowPassTs': float,        # Settling time to use for GPU temperature low pass filtering in units of epochs
	'GPUTempSafetyMargin': float,     # Safety margin for GPU overheating detection (if low-pass GPU temp exceeds this many degrees below hardware slowdown temperature then trigger overheating detection)
	'RunLockEnterDelay': float,       # Time interval to wait right after acquiring the lowest real run level lock or going solo
	'RunLockMaxCountsCPU': tuple,     # Maximum counts to use for device run level locking if the device is of type CPU
	'RunLockMaxCountsGPU': tuple,     # Maximum counts to use for device run level locking if the device is of type GPU
	'TrainEpochs': int,               # Number of epochs to train
	'TrainPerf': bool,                # Whether to evaluate the performance of model(s) resulting from training
	'TrainPerfAllSaved': bool,        # Whether to evaluate the performance of all (still existing) saved models or just the last one
	'TrainPerfOptimise': bool,        # Whether to optimise threshold hyperparameters in order to find the best possible model performance
	'TrainLockFirstEpochs': int,      # Number of first epochs to apply a high memory high execution solo lock to (-1 => Solo lock all epochs, 0 => Disable solo locking, 1 => Solo lock up to end of first epoch, ...)
	'TrainLogCSV': bool,              # Whether to log training results per epoch in CSV format
	'TrainResultsJSON': bool,         # Whether to save training results in JSON format
	'TrainResultsPickle': bool,       # Whether to save training results in pickle format
	'TrainResultsYAML': bool,         # Whether to save training results in YAML format
	'TrainRefLosses': bool,           # Whether to calculate reference losses in order to give model losses a sense of scale
	'TrainSaveNumLatest': int,        # Maximum number of model saves to keep in the models save directory (always keeping the latest, 0 = Keep all)
	'TrainSaveOnlyIfBeatsRef': bool,  # Only save new best models during training if they are better than all the reference losses
}

# Config parameter check specifications
# noinspection PyTypeChecker
pnn_config_check = {
	'GPUTempLowPassTs': 0,
	'GPUTempSafetyMargin': 0,
	'RunLockEnterDelay': 0,
	'RunLockMaxCountsCPU': [(len(device_util.GPULevel) - 2,) * 2, 1],
	'RunLockMaxCountsGPU': [(len(device_util.GPULevel) - 2,) * 2, 1],
	'TrainEpochs': 1,
	'TrainLockFirstEpochs': -1,
	'TrainSaveNumLatest': 0,
}

# File extensions
SAVED_MODEL_EXT = '.model'
SAVED_MODEL_META_EXT = '.yaml'
SAVED_MODEL_META_SUFFIX = '_meta'
SAVED_MODEL_META_SUFFIX_EXT = SAVED_MODEL_META_SUFFIX + SAVED_MODEL_META_EXT

# Print formats
perf_fmt = lambda value: 'd' if isinstance(value, int) else '#.4g'  # noqa

###############
### Classes ###
###############

# PNN manager class
class PNNManager:

	#
	# Construction
	#

	def __init__(self, name, version, script_path, config_spec, config_check, default_config_file=None, default_csvfmt_file=None, config_converters=None, config_kwargs=None):
		# name = Name of neural network, e.g. 'MyNet'
		# version = Version of neural network, e.g. '0.1'
		# script_path = Path of main script file, e.g. __file__ ('/path/to/my_net.py', relative or absolute)
		# config_spec = See config.ConfigManager class
		# config_check = See config.ConfigManager class
		# default_config_file = Default configuration file path, e.g. 'my_net.cfg' (default: script_path with extension changed to *.cfg)
		# default_csvfmt_file = Default CSV format file path, e.g. 'my_net_csvfmt.yaml' (default: script_path with extension changed to *_csvfmt.yaml)
		# config_converters = See config.ConfigManager class (converters argument of __init__)
		# config_kwargs = See config.ConfigManager class (kwargs of __init__)

		self.default_group = 'Default'
		self.default_models_name = 'models'
		self.models_subdir_meta = '.models_dir'
		self.timestamp_format = '%Y%m%d_%H%M%S'
		self.timestamp_frac_format = '.%f'
		self.git_force_valid_patch = False
		self.git_snapshot_tracked_binary = True
		self.git_snapshot_untracked_binary = False
		self.source_snapshot_builtins = False
		self.source_snapshot_in_prefix = False

		change_constants_fn = getattr(self, 'change_constants', None)
		if callable(change_constants_fn):
			change_constants_fn()

		self.name = name
		self.version = str(version)
		self.script_path = os.path.abspath(script_path)
		self.script_file = os.path.basename(self.script_path)
		self.script_dir_path = os.path.dirname(self.script_path)
		self.script_dir_name = os.path.basename(self.script_dir_path)

		gpu_index = -1
		match = re.search(r'GPU([0-9]+)$', self.script_dir_name)
		if match:
			try:
				gpu_index = int(match.group(1))
			except ValueError:
				raise ValueError(f"Found GPU specification in directory name '{self.script_dir_name}', but failed to extract desired default GPU index")

		self.default_cuda_device = f'cuda:{gpu_index}' if gpu_index >= 0 else 'auto'
		if gpu_index >= 0 or re.search(r'GPU[A-Z]$', self.script_dir_name):
			self.default_models_dir = os.path.abspath(os.path.join(self.script_dir_path, '..', self.default_models_name))
		else:
			self.default_models_dir = os.path.join(self.script_dir_path, self.default_models_name)
		self.default_config_file = default_config_file if default_config_file else os.path.splitext(self.script_path)[0] + '.cfg'
		self.default_csvfmt_file = default_csvfmt_file if default_csvfmt_file else os.path.splitext(self.script_path)[0] + '_csvfmt.yaml'
		self.config_spec = config_spec
		self.config_check = config_check
		self.config_converters = config_converters
		self.config_kwargs = {} if config_kwargs is None else config_kwargs

		self.device = None
		self.run_lock = None
		self.models_dir = None
		self.config_file = None
		self.config_manager = None
		self.csvfmt_file = None
		self.csvfmt_manager = None

		self.input_saved_model = None
		self.input_saved_model_list = [None]
		self.output_saved_model = None
		self.output_saved_model_list = [None]
		self.cm = {context: None for context in Context}

	def initialise(self, config_file=None, csvfmt_file=None, models_dir=None, device=None):
		# config_file = Configuration file path to use (None for default)
		# csvfmt_file = CSV format file path to use (None for default)
		# models_dir = Directory to load/save models from/to (None for default)
		# device = Torch device to use (see device_util.resolve_device() function, or 'auto')

		misc_util.print_header(self.__class__.__name__)

		if device is None:
			device = self.default_cuda_device
		if device == 'auto':
			self.device = torch.device(type='cuda', index=0) if torch.cuda.device_count() <= 1 else None
		else:
			self.device = device_util.resolve_device(device)
		print(f"Device: {'Will auto-select' if self.device is None else repr(self.device)}")
		self.run_lock = device_util.GPULevelLockV(self.device, (1, 1, 1, 1), lock_delay=1, autoselect_cb=self.device_autoselect_cb, verbose=True, newline=True)

		self.models_dir = os.path.abspath(models_dir if models_dir is not None else self.default_models_dir)
		models_dir_note = ''
		try:
			statinfo = os.stat(self.models_dir)
			if not stat.S_ISDIR(statinfo.st_mode):
				raise NotADirectoryError(f"Specified models directory exists but is not a directory: {self.models_dir}")
		except FileNotFoundError:
			models_dir_parent = os.path.dirname(self.models_dir)
			if not os.path.isdir(models_dir_parent):
				raise NotADirectoryError(f"Specified models directory does not exist and neither does its parent directory => Not allowing directory creation on demand for safety reasons: {self.models_dir}")
			models_dir_note = ' (WILL BE CREATED ON DEMAND)'
		print(f"Models directory{models_dir_note}: {self.models_dir}")

		print()

		self.config_file = os.path.abspath(config_file if config_file is not None else self.default_config_file)
		print(f"Config file: {self.config_file}")
		self.config_manager = config.ConfigManager(self.config_file, self.config_spec, config_check=self.config_check, converters=self.config_converters, **self.config_kwargs)
		print("Loaded configurations:")
		ppyutil.print.print_as_columns(self.config_manager.config_names(), line_prefix='  ')
		print()

		self.csvfmt_file = os.path.abspath(csvfmt_file if csvfmt_file is not None else self.default_csvfmt_file)
		print(f"CSV format file: {self.csvfmt_file}")
		self.csvfmt_manager = yaml_spec.YAMLSpecManager(self.csvfmt_file, join_lists=yaml_spec.JoinLists.Append)
		print("Loaded CSV formats:")
		ppyutil.print.print_as_columns(self.csvfmt_manager.spec_names(), line_prefix='  ')
		print()

		return self

	def device_autoselect_cb(self, selected_device):
		self.device = selected_device

	def ensure_models_dir_exists(self):
		try:
			os.mkdir(self.models_dir)
			print(f"Created models directory: {self.models_dir}")
		except OSError:
			# Cannot rely on checking for EEXIST, since the operating system could give priority to other errors like EACCES or EROFS
			if not os.path.isdir(self.models_dir):
				raise Exception(f"Failed to create required models directory (does the parent directory exist?): {self.models_dir}")
			print(f"Using models directory: {self.models_dir}")

	def create_cmdline_argparser(self):

		parser = argparse.ArgumentParser(
			prog=f"{self.name}",
			add_help=False,
			usage=f"{self.script_file} [OPTIONS] [ACTIONS]",
			description=f"Command line tool for training, running and testing the {self.name} network."
		)

		opt_general = parser.add_argument_group('General options')
		opt_general.add_argument('-h', '--help', action='help', help='Show this help message and exit')
		opt_general.add_argument('-v', '--version', action='version', version=f'{self.version}', help='Show version number (%(version)s) and exit')
		opt_general.add_argument('--group', dest='group', action='store', default=self.default_group, metavar='GROUP', help='Group to assign the current run to (default: %(default)s)')
		opt_general.add_argument('-c', '--config', dest='config', action='store', default=Configuration.Default, metavar='CONFIG', help='Configuration to use by default for completing actions with no explicit configuration specified')
		opt_general.add_argument('-f', '--csvfmt', dest='csvfmt', action='store', default=None, metavar='FORMAT', help='CSV format to use for outputting results data (default: use default CSV format)')
		opt_general.add_argument('-e', '--epochs', dest='epochs', action='store', type=IntRange(1), default=0, metavar='NUM', help='Number of epochs to train (default: use value from active configuration)')

		opt_initialise = parser.add_argument_group('Initialisation options')
		opt_initialise.add_argument('--config_file', dest='config_file', action='store', default=self.default_config_file, metavar='FILE', help='Config file to use (default: %(default)s)')
		opt_initialise.add_argument('--csvfmt_file', dest='csvfmt_file', action='store', default=self.default_csvfmt_file, metavar='FILE', help='CSV format file to use (default: %(default)s)')
		opt_initialise.add_argument('--models_dir', dest='models_dir', action='store', default=self.default_models_dir, metavar='DIR', help='Directory to load/save models from/to (default: %(default)s)')
		opt_initialise.add_argument('-d', '--device', '-g', dest='device', action='store', default=self.default_cuda_device, metavar='DEVICE', help='Device on which to run the network, e.g. auto, cpu, cuda, cuda:1, 2, 01:00.0, 0:A0 (default: %(default)s)')
		opt_initialise.add_argument('--cpu', dest='device', action='store_const', const='cpu', help='Shortcut for \'--device %(const)s\'')
		opt_initialise.add_argument('--cuda', dest='device', action='store_const', const=self.default_cuda_device, help='Shortcut for \'--device %(const)s\'')

		opt_action = parser.add_argument_group('Actions')
		opt_action.add_argument('--repeat', dest='repeat', action='store', type=IntRange(1), default=1, metavar='NUM', help='Repeat the entire action agenda a given number of times (default: %(default)d)')
		opt_action.add_argument('--debug_args', dest='agenda', key=Action.DebugArgs, action=AppendData, help='Debug the parsing of the command line arguments')
		opt_action.add_argument('--load_model', dest='agenda', key=Action.LoadModel, action=AppendData, nargs=1, metavar='MODEL', help='Load a model as the starting point for the next action (File => Find model file in models directory, Path => Explicit path to model file)')
		opt_action.add_argument('--keep_model', dest='agenda', key=Action.KeepModel, action=AppendData, help='Keep the model resulting from the last generative action as the starting point for the next action')
		opt_action.add_argument('--reset_model', dest='agenda', key=Action.ResetModel, action=AppendData, help='Clear any stored starting point model')
		opt_action.add_argument('--show_configs', dest='agenda', key=Action.ShowConfigs, action=AppendData, help='Show all available configurations and respective config parameters')
		opt_action.add_argument('--show_csvfmts', dest='agenda', key=Action.ShowCSVFmts, action=AppendData, help='Show details of all available CSV formats')
		opt_action.add_argument('--git_snapshot', dest='agenda', key=Action.GitSnapshot, action=AppendData, nargs='?', metavar='PATCH', help='Provide help in temporarily resetting the code to a particular git snapshot, e.g. so that a saved model can be loaded correctly')
		opt_action.add_argument('--model_info', dest='agenda', key=Action.ModelInfo, action=AppendData, nargs='?', metavar='CONFIG', help='Show information about the neural network model')
		opt_action.add_argument('--draw_model', dest='agenda', key=Action.DrawModel, action=AppendData, nargs=1, metavar='PDFPATH', help='Draw the neural network model to a pdf')
		opt_action.add_argument('--dataset_info', dest='agenda', key=Action.DatasetInfo, action=AppendData, help='Load a dataset and show how many samples there are')
		opt_action.add_argument('--dataset_stats', dest='agenda', key=Action.DatasetStats, action=AppendData, nargs='?', metavar='BASIS', help='Calculate statistics of the dataset (basis can be element, sample (default) or batch)')
		opt_action.add_argument('--train', dest='agenda', key=Action.Train, action=AppendData, nargs='*', metavar='CONFIG', help='Train the network in the given configurations (special values: nondefault, all)')
		opt_action.add_argument('--train_all', dest='agenda', key=Action.TrainAll, action=AppendData, nargs='?', metavar='STR', help='Train the network in all configurations starting with the specified string (special values: nondefault, all)')
		opt_action.add_argument('--perf_model', dest='agenda', key=Action.PerfModel, action=AppendData, nargs='*', metavar='MODEL', help='Evaluate the performance of the specified models (specified by file name or path)')
		opt_action.add_argument('--perf_model_optim', dest='agenda', key=Action.PerfModelOptim, action=AppendData, nargs='*', metavar='MODEL', help='Same as --perf_model but force parameter optimisation')

		return parser, opt_general, opt_initialise, opt_action

	def parse_arguments(self, argv):

		parser = self.create_cmdline_argparser()[0]
		args = parser.parse_args(argv)

		if not args.agenda:
			args.agenda = []

		with contextlib.suppress(ValueError):
			args.device = int(args.device)

		return args

	#
	# Run command line
	#

	def run_cmdline(self, argv):
		# argv = List of arguments to parse, e.g. sys.argv[1:]

		args = self.parse_arguments(argv)

		ppyutil.print.printc(f"{self.name} {self.version}", misc_util.header_color)
		print()

		self.initialise(config_file=args.config_file, csvfmt_file=args.csvfmt_file, models_dir=args.models_dir, device=args.device)

		with self.enter_cm(Context.Process):

			for repetition in range(args.repeat):
				for item in args.agenda:
					with self.enter_cm(Context.Action):

						if isinstance(item, str):
							key = item
							arg = None
						else:
							key = item[0]
							arg = item[1]

						if not self.handle_action(key, arg, args):
							misc_util.print_header("Unknown agenda key")
							print_warn(f"Unknown agenda key '{key}'")
							print()

			misc_util.print_header("All actions completed")

		return self

	def handle_action(self, key, arg, args):
		# key = Action key (see 'key=' in create_cmdline_argparser)
		# arg = Arguments passed to key on the command line
		# args = All parsed command line arguments

		if key == Action.DebugArgs:
			misc_util.print_header("Debug command line arguments")
			print('Parsed arguments:')
			ppyutil.print.pprint_to_width(vars(args))
			print()

		elif key == Action.LoadModel:
			self.load_saved_model(saved_model=arg[0])

		elif key == Action.KeepModel:
			self.keep_saved_model()

		elif key == Action.ResetModel:
			self.reset_saved_model()

		elif key == Action.ShowConfigs:
			self.show_configs()

		elif key == Action.ShowCSVFmts:
			self.show_csvfmts()

		elif key == Action.GitSnapshot:
			self.git_snapshot(patch_path=arg)

		elif key == Action.ModelInfo:
			self.model_info(configuration=args.config if arg is None else arg)

		elif key == Action.DrawModel:
			self.draw_model(configuration=args.config, pdfpath=arg[0])

		elif key == Action.DatasetInfo:
			self.dataset_stats(configuration=args.config, stats_basis=arg, quick=True)

		elif key == Action.DatasetStats:
			self.dataset_stats(configuration=args.config, stats_basis=arg, quick=False)

		elif key == Action.Train:
			train_kwargs = {'csvfmt': args.csvfmt, 'group': args.group, 'epochs': args.epochs}
			if not arg:
				self.train_network(configuration=args.config, **train_kwargs)
			elif len(arg) == 1:
				configuration = arg[0]
				if configuration == 'nondefault':
					self.train_networks(configurations=Configuration.NonDefault, **train_kwargs)
				elif configuration == 'all':
					self.train_networks(configurations=Configuration.All, **train_kwargs)
				else:
					self.train_network(configuration=configuration, **train_kwargs)
			else:
				self.train_networks(configurations=arg, **train_kwargs)

		elif key == Action.TrainAll:
			train_kwargs = {'csvfmt': args.csvfmt, 'group': args.group, 'epochs': args.epochs}
			if not arg or arg == 'all':
				self.train_networks(configurations=Configuration.All, **train_kwargs)
			elif arg == 'nondefault':
				self.train_networks(configurations=Configuration.NonDefault, **train_kwargs)
			else:
				self.train_networks(configurations=lambda cname: cname.startswith(arg), **train_kwargs)

		elif key == Action.PerfModel:
			self.eval_model_perfs(configuration=args.config, saved_models=arg, force_optim=False)

		elif key == Action.PerfModelOptim:
			self.eval_model_perfs(configuration=args.config, saved_models=arg, force_optim=True)

		else:
			return False

		return True

	#
	# Context management
	#

	@contextlib.contextmanager
	def enter_cm(self, context):  # Note: Process context is REQUIRED, must envelop ALL action contexts, and the process MUST exit/quit right after leaving the process context manager (otherwise it could illegally continue to take up CUDA memory for example)
		# context = Context to enter (Context enum)
		# Return the dynamic context instance used to enter the required context

		if context not in self.cm:
			raise ValueError(f"Unknown context: {context}")

		if self.cm[context]:  # We implement 'reentrance' by simply doing nothing if we detect we are already inside a 'context' context
			yield self.cm[context]
			return

		def clear_context_callback():
			self.cm[context] = None

		newline_config = self.run_lock.config(newline=False, block_newline=True, restore_to_init=True, sticky=True)
		with contextlib.ExitStack() as stack:
			with ppyutil.contextman.DynamicContext() as cm:
				cm.register_callback(clear_context_callback)
				self.cm[context] = cm
				if context == Context.Process:
					cm.enter_context(self.run_lock, key="run_lock")
				try:
					yield cm
				finally:
					stack.enter_context(newline_config)
					if context == Context.Process:
						self.process_cleanup()
					elif context == Context.Action:
						self.action_cleanup()

	def process_cleanup(self):
		# Clean up whatever should always be cleaned right before the Process context exits (before anything is popped from the associated exit stack), no matter what happened inside the Process context
		self.action_cleanup()  # In case no Action context was entered during the Process context, we treat the entire Process context as a single Action and clean up after that Action now. If Action contexts WERE entered, this justs redundantly cleans up again after the last action, which is okay too.

	# noinspection PyUnresolvedReferences
	def action_cleanup(self):
		# Clean up whatever should always be cleaned right before the Action context exits (before anything is popped from the associated exit stack), no matter what happened inside the Action context
		if torch.cuda.is_initialized():
			if self.device is None:
				print_warn("CUDA runtime is in the initialised state even though no device has been selected => This should not happen...")
				print()
			elif self.device.type == 'cuda':
				torch.cuda.synchronize(self.device)
			else:
				print_warn(f"CUDA runtime is in the initialised state even though the current device is not a CUDA device ({self.device.type}) => This should not happen...")
				print()
		gc.collect()
		if torch.cuda.is_initialized():
			torch.cuda.ipc_collect()
			torch.cuda.empty_cache()
		self.verify_lowmem()

	def ensure_entered(self, context):
		# context = Context to ensure has been entered (Context enum)
		# Return the associated dynamic context instance
		if context not in self.cm:
			raise ValueError(f"Unknown context: {context}")
		context_cm = self.cm[context]
		if not context_cm:
			raise ValueError(f"Specified context has not been entered yet: {context}")
		return context_cm

	def ensure_lowmem(self):
		process_cm = self.ensure_entered(Context.Process)
		if "run_lock_lowmem" not in process_cm:
			process_cm.enter_context(self.run_lock.level(GPULevel.LowMemNoExec), key="run_lock_lowmem", parent="run_lock")

	def clear_lowmem(self):
		process_cm = self.ensure_entered(Context.Process)
		if "run_lock_lowmem" in process_cm:
			process_cm.leave_context("run_lock_lowmem")

	# noinspection PyUnresolvedReferences
	def verify_lowmem(self):
		if ppyutil.execlock.process_exiting():
			return
		if torch.cuda.is_initialized():
			if sys.exc_info()[0] is None:
				self.ensure_lowmem()  # Fail-safe against bad code that doesn't call ensure_lowmem() before executing tasks that can initialise the CUDA runtime and allocate CUDA memory
		else:
			self.clear_lowmem()  # Although ensure_lowmem() was called, the CUDA runtime was not subsequently initialised, so no CUDA memory has been allocated yet

	#
	# Configurations
	#

	def resolve_configuration_list(self, configurations: Union[Configuration, Callable[[str], bool], Iterable[Union[Configuration, str, config.Config]]]):
		# configurations = Configuration enum, callable (str -> bool), or iterable of configuration specifications accepted by resolve_configuration()
		# Return list of configurations of type config.Config

		if configurations == Configuration.Default:
			configurations = [self.config_manager.default_name()]
		elif configurations == Configuration.NonDefault:
			default_cname = self.config_manager.default_name()
			configurations = [C for cname, C in self.config_manager.config_dict().items() if cname != default_cname]
		elif configurations == Configuration.All:
			configurations = list(self.config_manager.config_dict().values())
		elif callable(configurations):
			configurations = [C for cname, C in self.config_manager.config_dict().items() if configurations(cname)]

		return [self.resolve_configuration(configuration) for configuration in configurations]

	def resolve_configuration(self, configuration: Union[Configuration, str, config.Config]) -> Any:
		# configuration = Configuration.Default, str or config.Config
		# Return configuration of type config.Config

		if configuration == Configuration.Default:
			C = self.config_manager.default_config()
		elif isinstance(configuration, str):
			C = self.config_manager.get_config(configuration)
		else:
			C = configuration

		if not isinstance(C, config.Config):
			raise TypeError(f"Invalid single configuration (should be of type config.Config): {C}")

		return C

	def resolve_csvfmt(self, csvfmt):
		# csvfmt = None (Default format), str or yaml_spec.YAMLSpec
		# Return CSV format of type yaml_spec.YAMLSpec

		if csvfmt is None:
			CSVF = self.csvfmt_manager.default_spec()
		elif isinstance(csvfmt, str):
			CSVF = self.csvfmt_manager.get_spec(csvfmt)
		else:
			CSVF = csvfmt

		if not isinstance(CSVF, yaml_spec.YAMLSpec):
			raise TypeError(f"Invalid CSV format object (should be of type yaml_spec.YAMLSpec): {CSVF}")

		return CSVF

	def resolve_saved_model(self, saved_model):
		# saved_model = Saved model specification to resolve (None => Use raw model, File => Find model file in models directory, Path => Explicit path to model file)
		# Return absolute saved model path or None or raise an error if not found
		if saved_model is None:
			return None
		saved_model_path = os.path.abspath(saved_model)
		if not os.path.isfile(saved_model_path) and os.path.basename(saved_model) == saved_model:
			saved_model_path = None
			alternative_paths = set()
			for dirpath, dirnames, filenames in os.walk(self.models_dir, topdown=False, followlinks=False):
				if saved_model in filenames:
					new_saved_model_path = os.path.join(dirpath, saved_model)
					if saved_model_path is None or saved_model_path < new_saved_model_path:
						if saved_model_path is not None:
							alternative_paths.add(saved_model_path)
						saved_model_path = new_saved_model_path
					else:
						alternative_paths.add(new_saved_model_path)
			for model_path in sorted(alternative_paths):
				print_warn(f"Ambiguous alternative saved model: {model_path}")
		if saved_model_path is None:
			raise FileNotFoundError(f"Failed to find saved model: {saved_model}")
		return saved_model_path

	@staticmethod
	def resolve_stats_basis(stats_basis):
		# stats_basis = None (Default basis) or str
		# Return a valid DatasetStatsBasis
		if stats_basis is None:
			return DatasetStatsBasis.Default
		elif isinstance(stats_basis, str):
			return DatasetStatsBasis.from_str(stats_basis)
		else:
			raise TypeError(f"Invalid dataset statistics basis specification (should be None or str): {stats_basis}")

	#
	# Actions
	#

	def load_saved_model(self, saved_model):
		misc_util.print_header("Load saved model")
		print(f"Saved model to load:  {saved_model}")
		saved_model = self.resolve_saved_model(saved_model)
		print(f"Resolved saved model: {saved_model}")
		self.input_saved_model = saved_model
		self.input_saved_model_list = [saved_model]
		print()
		self.show_input_saved_models()

	def keep_saved_model(self):
		misc_util.print_header("Keep saved model")
		print("Keeping current saved model(s) as the starting point for the next action...")
		self.input_saved_model = self.output_saved_model
		self.input_saved_model_list = self.output_saved_model_list
		print()
		self.show_input_saved_models()

	def reset_saved_model(self):
		misc_util.print_header("Reset saved model")
		print("Resetting current saved model(s) for the next action...")
		self.input_saved_model = None
		self.input_saved_model_list = [None]
		print()
		self.show_input_saved_models()

	def show_input_saved_models(self):
		print("Main saved model:")
		print(f"  {self.input_saved_model}")
		print()
		print("Saved model list:")
		for saved_model in self.input_saved_model_list:
			print(f"  {saved_model}")
		print()

	def show_configs(self):
		misc_util.print_header("Show available configurations")
		print(f"Config file: {self.config_file}")
		print("Loaded configurations:")
		ppyutil.print.print_as_columns(self.config_manager.config_names(), line_prefix='  ')
		print()
		self.config_manager.pprint()

	def show_csvfmts(self):
		misc_util.print_header("Show available CSV formats")
		print(f"CSV format file: {self.csvfmt_file}")
		print("Loaded CSV formats:")
		ppyutil.print.print_as_columns(self.csvfmt_manager.spec_names(), line_prefix='  ')
		print()
		self.csvfmt_manager.pprint(header='CSV format')

	def git_snapshot(self, patch_path=None):
		# patch_path = A specific git snapshot patch file to adjust the provided helpful commands to

		misc_util.print_header("Git snapshot help")

		use_patch = False
		empty_patch = True
		patch_path_abs = None
		repo_path = None
		commit_hash = None
		tracked_binary = None
		untracked_binary = None

		if patch_path is None:

			print("Providing general help, as no specific patch file was passed as an argument...")
			print()

		else:

			patch_path_abs = os.path.abspath(patch_path)
			print(f"Providing help for: {patch_path_abs}")

			try:

				with open(patch_path_abs, 'r') as file:
					for line in file:
						if line.startswith('diff') or (line.startswith('@@') and line.rstrip() != '@@ -1,1 +1,1 @@ empty_diff'):
							empty_patch = False
							break
						elif line.startswith('Git repo:'):
							words = line.split()
							if len(words) >= 3:
								repo_path = words[2]
						elif line.startswith('Latest commit:'):
							words = line.split()
							if len(words) >= 3:
								commit_hash = words[2]
						elif line.startswith('Includes tracked binary files:'):
							with contextlib.suppress(ValueError, IndexError):
								tracked_binary = strtobool(line.split()[4])
						elif line.startswith('Includes untracked binary files:'):
							with contextlib.suppress(ValueError, IndexError):
								untracked_binary = strtobool(line.split()[4])

				if repo_path:
					print(f"Repo path: {repo_path}")
				else:
					print_warn("Failed to parse git repository path from patch file")

				if commit_hash:
					print(f"Commit hash: {commit_hash}")
				else:
					print_warn("Failed to parse required commit hash from patch file")

				use_patch = bool(repo_path and commit_hash)

				if tracked_binary is None:
					print_warn("Failed to parse whether tracked binary files are included in the patch file")
				if untracked_binary is None:
					print_warn("Failed to parse whether untracked binary files are included in the patch file")

			except OSError:
				print_warn("Failed to open/read specified patch file => Ignoring patch file and providing only general help instead...")

			print()

		have_notes = False

		repo_dirty = True
		orig_head = None
		if use_patch:
			repo = ppyutil.git.get_git_repo(path=repo_path)
			if repo is None:
				print_warn(f"Failed to open git repository => Skipping checks that require repository access...")
				have_notes = True
			else:
				repo_dirty = repo.is_dirty(index=True, working_tree=True, untracked_files=True)
				orig_head = ppyutil.git.head_symbolic_ref(repo)
			if not repo_dirty:
				print("Note: Git repository is currently clean (no working changes/untracked files) => Skipping git stash/pop in the instructions below...")
				have_notes = True

		if use_patch and empty_patch:
			print("Note: Patch file contains no actual working changes => Skipping git apply in the instructions below...")
			have_notes = True

		if not tracked_binary or not untracked_binary:
			if tracked_binary is None:
				print(f"Note: Tracked binary files may or may not have been included in the patch file (probably {'yes' if self.git_snapshot_tracked_binary else 'no'}) => Take care!")
			elif not tracked_binary:
				print(f"Note: Patch file does not include tracked binary files => Take care!")
			if untracked_binary is None:
				print(f"Note: Untracked binary files may or may not have been included in the patch file (probably {'yes' if self.git_snapshot_untracked_binary else 'no'}) => Take care!")
			elif not untracked_binary:
				print(f"Note: Patch file does not include untracked binary files => Take care!")
			have_notes = True

		if have_notes:
			print()

		ind = '  '
		step = 0

		if not use_patch or repo_dirty:
			step += 1
			print(f"{step}) Get the git repo into a clean state (no working changes/untracked files):")
			print(f"   {ind}cd \"{repo_path if use_patch else 'REPO_PATH'}\"")
			print(f"   {ind}git stash -u")
			print("   If there were no local changes to save, no stash is created and you shouldn't attempt to pop the stash again at the end.")
			print()

		step += 1
		if orig_head:
			print(f"{step}) Check out the required snapshot commit:")
		else:
			print(f"{step}) Save the current git HEAD and proceed to check out the required snapshot commit:")
			print(f"   {ind}ORIGHEAD=\"$(git symbolic-ref -q --short HEAD || git rev-parse HEAD)\" && echo \"$ORIGHEAD\"")
		print(f"   {ind}git checkout {commit_hash if use_patch else 'COMMIT_HASH'}")
		print()

		if not use_patch or not empty_patch:
			step += 1
			print(f"{step}) Apply the snapshot patch:")
			print(f"   {ind}git apply \"{patch_path_abs if use_patch else 'PATCH_PATH'}\"")
			print()

		step += 1
		print(f"{step}) Do whatever you want to do at this snapshot, e.g. test a model that was saved at this snapshot.")
		print("   Just note that if you add/change any git-ignored files, then these changes will remain when returning to the original HEAD state.")
		print("   Staged, unstaged and untracked files are not a problem however, and are managed/restored correctly.")
		print()

		step += 1
		print(f"{step}) Once done, if you made any non-git-ignored changes then you need to deal with them.")
		print("   If you have any changes that you want to keep for accessing again some other time:")
		print(f"   {ind}git checkout -b NEW_BRANCH")
		print(f"   {ind}git gui")
		print(f"   {ind}# <-- Commit the changes you want to keep")
		print("   Push the new branch to the remote if desired (sets up branch tracking):")
		print(f"   {ind}git push -u origin NEW_BRANCH")
		print("   Discard any remaining working directory changes:")
		print(f"      {ind}git reset --hard HEAD")
		print(f"      {ind}git clean -df")
		print()

		step += 1
		print(f"{step}) Return the git repo to the original HEAD state:")
		if orig_head:
			print(f"   {ind}git checkout {orig_head}")
		else:
			print(f"   {ind}git checkout \"$ORIGHEAD\"")
		print()

		if not use_patch or repo_dirty:
			step += 1
			print(f"{step}) Restore the original working changes and untracked files (ONLY if a stash was actually created at the beginning):")
			print(f"   {ind}git stash pop")
			print()

		print("Note that if you train and save a model while at the git snapshot, the saved model file(s) will still be available when returning to the original git HEAD as they are git-ignored. This works the same for all other git-ignored files as well.")
		print()

	def model_info(self, configuration=Configuration.Default):
		# configuration = Configuration (see resolve_configuration() function)
		misc_util.print_header("Model info")
		C = self.resolve_configuration(configuration)
		self.model_info_impl(C)

	def draw_model(self, configuration=Configuration.Default, pdfpath=None):
		# configuration = Configuration (see resolve_configuration() function)
		# pdfpath = String path of the required output pdf file (None => Just return the generated graph)
		misc_util.print_header("Draw model")
		C = self.resolve_configuration(configuration)
		return self.draw_model_impl(C, pdfpath=pdfpath)

	def dataset_stats(self, configuration=Configuration.Default, stats_basis=None, quick=False):
		# configuration = Configuration (see resolve_configuration() function)
		# stats_basis = Dataset statistics basis (see resolve_stats_basis() function)
		# quick = Whether to avoid long running operations
		misc_util.print_header("Dataset statistics")
		C = self.resolve_configuration(configuration)
		basis = self.resolve_stats_basis(stats_basis)
		return self.dataset_stats_impl(C, basis=basis, quick=quick)

	def train_networks(self, configurations: Any = Configuration.All, csvfmt=None, group=None, epochs=None):
		# configurations = Configuration list (see resolve_configuration_list() function)
		# csvfmt = CSV format to use (see resolve_csvfmt() function)
		# group = Group to assign this training run to
		# epochs = Maximum number of epochs to train (overrides configurations if >= 1)
		# Return list of individual train_network() return values

		misc_util.print_header("Train network in multiple configurations")

		configurations = self.resolve_configuration_list(configurations)
		CSVF = self.resolve_csvfmt(csvfmt)

		print("Configurations to train:")
		for configuration in configurations:
			print(f"  {configuration.name()}")
		if not configurations:
			print_warn(f"Empty configuration list => Nothing to do")
		print()

		print("Additional options:")
		print(f"  CSV format: {CSVF.name}")
		if group is not None:
			print(f"  Group: {group}")
		if epochs is not None and epochs >= 1:
			print(f"  Epoch limit: {epochs}")
		print()

		ret = []
		for configuration in configurations:
			ret.append(self.train_network(configuration=configuration, csvfmt=CSVF, group=group, epochs=epochs))

		return ret

	def train_network(self, configuration: Union[Configuration, str, config.Config] = Configuration.Default, csvfmt=None, group=None, epochs=None):
		# configuration = Configuration (see resolve_configuration() function)
		# csvfmt = CSV format to use (see resolve_csvfmt() function)
		# group = Group to assign this training run to
		# epochs = Maximum number of epochs to train (overrides configuration if >= 1)
		# Return train_network_impl() return value
		misc_util.print_header("Train network")
		C = self.resolve_configuration(configuration)
		CSVF = self.resolve_csvfmt(csvfmt)
		return self.train_network_impl(C, CSVF=CSVF, group=group, epochs=epochs)

	def eval_model_perfs(self, configuration=Configuration.Default, saved_models=None, force_optim=False):
		# configuration = Configuration (see resolve_configuration() function)
		# saved_models = Sequence of saved models (see resolve_saved_model() function)
		# force_optim = Whether to force optimisation of performance parameters even if already optimised ones are already available
		# Return eval_model_perfs_impl() return value
		misc_util.print_header("Evaluate model performances")
		C = self.resolve_configuration(configuration)
		if saved_models:
			saved_model_paths = [self.resolve_saved_model(saved_model) for saved_model in saved_models]
		else:
			saved_model_paths = list(saved_model_path for saved_model_path in self.input_saved_model_list if saved_model_path is not None)[::-1]
		return self.eval_model_perfs_impl(C, saved_model_paths, force_optim=force_optim)

	#
	# Load network
	#

	# Note: MUST be called at the very beginning of every action (important amongst other things for determinism and run level locking)
	def apply_global_config(self, C: Any, showC=True, force_cpu=False, allow_cudnn_bench=True):
		# C = Configuration of type config.Config
		# showC = Whether to pretty print the value of C
		# force_cpu = Whether to force use only of the CPU (avoid CUDA)
		# allow_cudnn_bench = Whether to allow enabling of cuDNN benchmark mode
		# Return the system information gathered using system_info.print_system_info_summary()

		print(f"Running python script path: {self.script_path}")
		print()

		device_is_cpu = (self.device and self.device.type == 'cpu')
		max_counts_tuple = C.RunLockMaxCountsCPU if device_is_cpu else C.RunLockMaxCountsGPU
		self.run_lock.update_max_counts(dict(zip(self.run_lock.run_levels(), max_counts_tuple)))
		self.run_lock.lock_delay = C.RunLockEnterDelay

		if not force_cpu:
			self.ensure_lowmem()

		cpu_only = force_cpu or device_is_cpu
		if not cpu_only and self.device.index is not None:
			torch.cuda.set_device(self.device)

		print("Applying global configurations:")
		print(f"  Making calculations {'deterministic' if C.Deterministic else 'indeterministic'}")
		misc_util.update_determinism(C.Deterministic, C.DeterministicSeed, allow_cudnn_bench and not force_cpu and C.CudnnBenchmarkMode)
		# noinspection PyUnresolvedReferences
		print(f"  Cudnn benchmark mode: {torch.backends.cudnn.benchmark}")
		print()

		sysinfo = system_info.print_system_info_summary(cpu_only=cpu_only)

		if showC:
			C.pprint()

		return sysinfo

	def load_network(self, C, load_model_opts=None, load_dataset_opts=None, force_cpu=False):
		# C = Configuration of type config.Config
		# load_model_opts = Custom options for loading the model (e.g. for optional initialisation of model from state-dict object, or for multi-stage training where the model can change between stages in a way not implementable by simple changes to C)
		# load_dataset_opts = Custom options for loading the dataset (e.g. for multi-stage training where the dataset format can change between stages in a way not implementable by simple changes to C)
		# force_cpu = Whether to force the model to run on the CPU
		# Return model (netmodel.NetModel), data_loaders (dataset.DatasetTuple of torch.utils.data.DataLoader), datasets (dataset.DatasetTuple of dataset.StagedDataset), info dict about the loaded model, info dict about the loaded dataset, model run level lock or None, model C, model metadata dict

		model, model_info, model_lock, modelC, model_meta = self.load_network_model(C, load_model_opts, force_cpu=force_cpu)
		data_loaders, datasets, dataset_info = self.load_network_dataset(C, model.reqd_inputs, model.reqd_targets, model.device, load_dataset_opts)

		return model, data_loaders, datasets, model_info, dataset_info, model_lock, modelC, model_meta

	def load_network_model(self, C, load_model_opts, force_cpu=False):
		# C = Configuration of type config.Config
		# load_model_opts = Custom options for loading the model
		# force_cpu = Whether to force the model to run on the CPU
		# Return model (netmodel.NetModel), info dict about the loaded model, optional model run level lock, model C, model metadata dict (non-trivial if loaded from saved model)

		model_info: Dict[str, Any] = {}

		if force_cpu:
			model_lock = None
		else:
			action_cm = self.ensure_entered(Context.Action)
			model_lock = action_cm.enter_context(self.run_lock.level(GPULevel.NormalMemLowExec))

		print("Loading model:")
		model = self.load_model(C, load_model_opts)
		model_info['name'] = model.__class__.__name__
		print(f"  Loaded {model.__class__.__name__} model")
		if not isinstance(model, netmodel.NetModel):
			raise TypeError(f"Loaded model should be a subclass of netmodel.NetModel: {model.__class__}")

		if isinstance(load_model_opts, dict):
			load_saved_model = load_model_opts.get('load_saved_model', True)
			if load_saved_model is False:
				load_model_path = None
			elif load_saved_model is True:
				load_model_path = self.input_saved_model
			else:
				load_model_path = self.resolve_saved_model(load_saved_model)
		else:
			load_model_path = self.input_saved_model

		if load_model_path is not None:
			print(f"  Loading saved model state: {load_model_path}")
			modelC, model_meta, model_meta_path, model_meta_exists = self.load_model_state(model, load_model_path, load_meta=True)
			if model_meta_exists:
				print(f"  Associated metadata YAML:  {model_meta_path}")
			print(f"  Saved model configuration: {modelC.name()}")
			if model_meta:
				print("  Saved model metadata:")
				for meta_key, meta_value in sorted(model_meta.items()):
					print(f"    {meta_key} = {meta_value}")
		else:
			modelC = C
			model_meta = {}

		model_device = torch.device('cpu') if force_cpu else self.device
		print(f"  Moving model to {repr(model_device)}")
		model.to(model_device)
		if model_device.type == 'cpu':
			model_info['device'] = str(model_device)
			model_info['device_name'] = 'CPU'
			model_info['gpu'] = None
			print(f"  Model is running on CPU{' (forced)' if force_cpu and self.device.type != 'cpu' else ''}")
		else:
			device_index = device_util.resolve_device_index(model_device, enforce_type='cuda')
			device_name = torch.cuda.get_device_name(model_device)
			device_capab = torch.cuda.get_device_capability(model_device)
			model_info['device'] = f'{model_device.type}:{device_index}'
			model_info['device_name'] = device_name
			model_info['gpu'] = device_index
			print(f"  Model is running on GPU {device_index}: {device_name} (capability {device_capab[0]}.{device_capab[1]})")

		model_info['inputs'] = '|'.join('+'.join(model.reqd_inputs.channels[c].type.name for c in g) for g in model.reqd_inputs.groups)
		print(f"  Input channels = {', '.join(channel.type.name for channel in model.reqd_inputs.channels)}")
		for c, channel in enumerate(model.reqd_inputs.channels):
			channel_opts = f"{{{', '.join(f'{opt.name}: {value}' for opt, value in channel.opts.items())}}}"
			print(f"    Channel {c}: Options for {channel.type.name} = {channel_opts}")
		for g, group in enumerate(model.reqd_inputs.groups):
			print(f"  Input component {g} = Channel{'s' if len(group) > 1 else ''} {', '.join(str(c) for c in group)} = {', '.join(model.reqd_inputs.channels[c].type.name for c in group)}")

		model_info['targets'] = '|'.join('+'.join(model.reqd_targets.channels[c].type.name for c in g) for g in model.reqd_targets.groups)
		print(f"  Target channels = {', '.join(channel.type.name for channel in model.reqd_targets.channels)}")
		for c, channel in enumerate(model.reqd_targets.channels):
			channel_opts = f"{{{', '.join(f'{opt.name}: {value}' for opt, value in channel.opts.items())}}}"
			print(f"    Channel {c}: Options for {channel.type.name} = {channel_opts}")
		for g, group in enumerate(model.reqd_targets.groups):
			print(f"  Target component {g} = Channel{'s' if len(group) > 1 else ''} {', '.join(str(c) for c in group)} = {', '.join(model.reqd_targets.channels[c].type.name for c in group)}")

		model_info['outputs'] = '|'.join('+'.join(model.reqd_outputs.channels[c].type.name for c in g) for g in model.reqd_outputs.groups)
		print(f"  Output channels = {', '.join(channel.type.name for channel in model.reqd_outputs.channels)}")
		for c, channel in enumerate(model.reqd_outputs.channels):
			channel_opts = f"{{{', '.join(f'{opt.name}: {value}' for opt, value in channel.opts.items())}}}"
			print(f"    Channel {c}: Options for {channel.type.name} = {channel_opts}")
		for g, group in enumerate(model.reqd_outputs.groups):
			print(f"  Output component {g} = Channel{'s' if len(group) > 1 else ''} {', '.join(str(c) for c in group)} = {', '.join(model.reqd_outputs.channels[c].type.name for c in group)}")

		print()

		print(f"Model parameters:")
		num_total = sum(p.numel() for p in model.parameters())
		num_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
		num_untrain = num_total - num_train
		model_info['params'] = num_total
		model_info['params_train'] = num_train
		model_info['params_notrain'] = num_untrain
		print(f"  Num untrainable = {num_untrain:,}")
		print(f"  Num trainable = {num_train:,}")
		print(f"  Num total = {num_total:,}")
		print()

		print(f"Neural network model components:")
		print(model)
		print()

		return model, model_info, model_lock, modelC, model_meta

	@classmethod
	def load_model_state(cls, model, path, load_meta=False):
		# model = Model to load the given saved state into
		# path = Path of the saved model file to retrieve the model state from
		# load_meta = Load the associated metadata YAML file as well
		# Return the loaded model C, metadata, absolute path to the associated metadata YAML file, and whether a metadata YAML file was actually found

		path = os.path.abspath(path)
		loaded_data = torch.load(path, map_location=torch.device('cpu'))

		modelC = loaded_data['C']
		if not isinstance(modelC, config.Config):
			raise ValueError(f"Loaded model C should be a config.Config, not: {type(modelC)}")
		elif modelC.name() != model.C.name():
			print_warn(f"Loaded model state was saved with configuration {modelC.name()} but current model class has configuration {model.C.name()} => This may not work")

		loaded_configs = modelC.dict()
		model_configs = model.C.dict()
		loaded_not_model_keys = sorted(loaded_configs.keys() - model_configs.keys())
		model_not_loaded_keys = sorted(model_configs.keys() - loaded_configs.keys())
		differing_items = sorted((key, loaded_configs[key], model_configs[key]) for key in loaded_configs.keys() & model_configs.keys() if loaded_configs[key] != model_configs[key])
		if loaded_not_model_keys:
			print_warn("Loaded model C has the following deprecated parameters:\n  {0}".format('\n  '.join(loaded_not_model_keys)))
		if model_not_loaded_keys:
			print_warn("Loaded model C did not have the following parameters:\n  {0}".format('\n  '.join(model_not_loaded_keys)))
		for key, lparam, mparam in differing_items:
			print_warn(f"Loaded model C and current model C disagree in parameter {key}: {lparam} vs {mparam}")

		model.load_state_dict(loaded_data['model_state_dict'])
		model.eval()

		if load_meta:
			model_dir, model_nameext = os.path.split(path)
			model_name = os.path.splitext(model_nameext)[0]
			default_model_meta_path = os.path.join(model_dir, re.sub(r'_[^_0-9]*$', SAVED_MODEL_META_SUFFIX_EXT, model_name, count=1))
			model_meta_paths = [
				os.path.join(model_dir, model_name + SAVED_MODEL_META_EXT),
				os.path.join(model_dir, model_name + SAVED_MODEL_META_SUFFIX_EXT),
				default_model_meta_path,
			]
			model_meta = {}
			for meta_path in model_meta_paths:
				try:
					with open(meta_path, 'r') as file:
						model_meta = yaml.load(file, Loader=yaml.CSafeLoader)
					if not isinstance(model_meta, dict):
						model_meta = {}
					model_meta_path = meta_path
					model_meta_exists = True
					break
				except OSError:
					continue
			else:
				model_meta_path = default_model_meta_path
				model_meta_exists = False
		else:
			model_meta = None
			model_meta_path = None
			model_meta_exists = False

		return modelC, model_meta, model_meta_path, model_meta_exists

	@classmethod
	def load_network_dataset(cls, C, reqd_inputs, reqd_targets, model_device, load_dataset_opts):
		# C = Configuration of type config.Config
		# reqd_inputs = Required model inputs (expects resolved dataset.DataSpec)
		# reqd_targets = Required model targets (expects resolved dataset.DataSpec)
		# model_device = Device on which the model that is going to use this dataset is located
		# load_dataset_opts = Custom options for loading the dataset
		# Return data_loaders (dataset.DatasetTuple of torch.utils.data.DataLoader), datasets (dataset.DatasetTuple of dataset.StagedDataset), info dict about the loaded dataset

		print(f"Loading dataset:")

		data_loaders, datasets, dataset_paths, dataset_details = cls.load_dataset(C, reqd_inputs, reqd_targets, model_device, load_dataset_opts)
		dataset_size = dataset.DatasetTuple(len(datasets.train), len(datasets.valid), len(datasets.test))
		total_size = dataset_size.train + dataset_size.valid + dataset_size.test

		dataset_info = {
			'path': dataset_paths,
			'size': dataset_size,
			'total_size': total_size,
		}

		if dataset_details:
			for detail_line in dataset_details:
				print('  ' + detail_line)

		print(f"  Training data:   {dataset_paths.train}")
		print(f"  Validation data: {dataset_paths.valid}")
		print(f"  Test data:       {dataset_paths.test}")
		print(f"  Training loader:   {data_loaders.train.batch_size} samples per batch, {data_loaders.train.num_workers} workers")
		print(f"  Validation loader: {data_loaders.valid.batch_size} samples per batch, {data_loaders.valid.num_workers} workers")
		print(f"  Test loader:       {data_loaders.test.batch_size} samples per batch, {data_loaders.test.num_workers} workers")
		print(f"  Loaded {dataset_size.train} training samples")
		print(f"  Loaded {dataset_size.valid} validation samples")
		print(f"  Loaded {dataset_size.test} test samples")
		print(f"  Total of {total_size} samples")
		print()

		return data_loaders, datasets, dataset_info

	@classmethod
	def load_model(cls, C, opts):
		# C = Configuration of type config.Config
		# opts = Custom options for loading the model (passed through from load_network function)
		# Return neural network model (must be a subclass of netmodel.NetModel)
		raise NotImplementedError(f"Class {cls.__name__} has not implemented the {inspect.currentframe().f_code.co_name}() function")

	@classmethod
	def load_dataset(cls, C, reqd_inputs, reqd_targets, model_device, opts):
		# C = Configuration of type config.Config
		# reqd_inputs = Required model inputs (expects resolved dataset.DataSpec)
		# reqd_targets = Required model targets (expects resolved dataset.DataSpec)
		# model_device = Device on which the model that is going to use this dataset is located
		# opts = Custom options for loading the dataset (passed through from load_network function)
		# Return data_loaders, datasets, dataset_paths (all are dataset.DatasetTuple of torch.utils.data.DataLoader, dataset.StagedDataset, str respectively)
		raise NotImplementedError(f"Class {cls.__name__} has not implemented the {inspect.currentframe().f_code.co_name}() function")

	@classmethod
	def load_criterion(cls, model):
		# model = Model to load a criterion for (netmodel.NetModel)
		# Return a criterion for the model (see netmodel.NetModel.get_criterion() function)
		print("Using criterion:")
		criterion = model.get_criterion_moved()
		print(ppyutil.string.add_line_prefix(str(criterion), '  '))
		print()
		return criterion

	@classmethod
	def load_optimizer(cls, model):
		# model = Model to optimize (netmodel.NetModel)
		# Return an optimizer for the model (see netmodel.NetModel.get_optimizer() function)
		print("Using optimizer:")
		optimizer = model.get_optimizer()
		print(ppyutil.string.add_line_prefix(str(optimizer), '  '))
		print()
		return optimizer

	@classmethod
	def load_scheduler(cls, C, optimizer, model):
		# C = Configuration of type config.Config
		# optimizer = Optimizer to schedule the learning rate of (torch.optim.Optimizer)
		# model = Model to which the learning rate scheduler will be applied (netmodel.NetModel)
		# Return a learning rate scheduler for the model (torch.optim.lr_scheduler._LRScheduler), and the required arguments for the scheduler (LRSchedulerArgs enum or iterable thereof)
		print("Using learning rate scheduler:")
		scheduler, reqd_scheduler_args = cls.load_scheduler_impl(C, optimizer, model)
		print(ppyutil.string.add_line_prefix(f"Scheduler object: {scheduler}", '  '))
		print(f"  Required arguments: {reqd_scheduler_args.name if isinstance(reqd_scheduler_args, LRSchedulerArgs) else ', '.join(sarg.name for sarg in reqd_scheduler_args)}")
		print()
		return scheduler, reqd_scheduler_args

	# noinspection PyUnusedLocal
	@classmethod
	def load_scheduler_impl(cls, C, optimizer, model) -> Tuple[Any, Union[LRSchedulerArgs, Iterable[LRSchedulerArgs]]]:
		# C = Configuration of type config.Config
		# optimizer = Optimizer to schedule the learning rate of (torch.optim.Optimizer)
		# model = Model to which the learning rate scheduler will be applied (netmodel.NetModel)
		# Return a learning rate scheduler for the model (torch.optim.lr_scheduler._LRScheduler), and the required arguments for the scheduler (LRSchedulerArgs enum or iterable thereof)
		# noinspection PyTypeChecker
		return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0), LRSchedulerArgs.NoArgs

	@classmethod
	def load_stopper(cls, C):
		# C = Configuration of type config.Config
		# Return a stopper class (training.StopperBase)
		print("Using stopper:")
		stopper = cls.load_stopper_impl(C)
		print(ppyutil.string.add_line_prefix(str(stopper), '  '))
		print()
		return stopper

	# noinspection PyUnusedLocal
	@classmethod
	def load_stopper_impl(cls, C):
		# C = Configuration of type config.Config
		# Return a stopper class (training.StopperBase)
		return training.StopperBase()

	#
	# Load/save model
	#

	# noinspection PyUnusedLocal
	@classmethod
	def get_model_save_data(cls, C, model, **kwargs):
		# C = Configuration of type config.Config
		# model = Model object to construct the save-data for
		# kwargs = Keyword arguments that can be used in overriders of this function to allow saving of more complicated/complete/custom states (e.g. to save the state during training in a way that can be continued later)
		# Return a Dict[Profile] of dicts containing all the information about a model that should be saved when saving to disk
		return {
			'cs': {'C': C, 'model_state_dict': model.state_dict(), 'model': model},
		}

	def create_models_subdir(self, dir_prefix, group, C=None, model=None, timestamp_prefix=None, timestamp=None, create_tee_logger=True, create_source_snapshot=True, create_git_snapshot=True, git_repo_spec=None, model_save_kwargs=None):
		# dir_prefix = Path prefix for the required timestamped subdirectory (interpreted relative to manager-wide models directory => MODELS_DIR/dir_prefix/group/timestamp_prefix_TIMESTAMP/MODELS)
		# group = Group to assign the required timestamped subdirectory to (see directory structure above)
		# C = Configuration of type config.Config (required if model is passed)
		# model = If only one model will be saved in this directory (but possibly many times, e.g. in different trained states), provide it here (gives default values for timestamp_prefix/git_repo_spec)
		# timestamp_prefix = String prefix to use for the subdirectory name (None = Derive from model, if available)
		# timestamp = Optional timestamp to use instead of just taking the current datetime
		# create_tee_logger = Whether to create a StdTee logger that can be used to log all stdout/stderr to file
		# create_source_snapshot = Whether to create a source snapshot in the new subdirectory if a model is provided (provides a summary of the source code of all the classes required to save the model)
		# create_git_snapshot = Whether to create a git snapshot in the new subdirectory (provides a snapshot of the current code state based on git)
		# git_repo_spec = Object/type that can be used to determine the location of the required git repository (None = Derive from model, if available)
		# model_save_kwargs = Keyword arguments to pass to internal calls to get_model_save_data()
		# Return the absolute path to the created models group directory, the absolute path to the created models subdirectory, the subdirectory name (basename of the previous return value), the timestamp that was used to create it, the StdTee logger that was created (if requested), the information that was retrieved about the git state (Dict or None), and the contents of the created source snapshot

		if not dir_prefix or os.path.isabs(dir_prefix):
			raise ValueError(f"Models subdirectory prefix cannot be empty or an absolute path (for safety reasons): {dir_prefix}")
		if not group or os.path.isabs(group):
			raise ValueError(f"Models subdirectory group cannot be empty or an absolute path (for safety reasons): {group}")

		if model is not None and C is None:
			raise ValueError(f"Configuration C is required if a model is passed")
		if model_save_kwargs is None:
			model_save_kwargs = {}

		if timestamp is None:
			timestamp = datetime.datetime.now()

		if model is not None:
			if timestamp_prefix is None:
				timestamp_prefix = model.__class__.__name__
			if create_git_snapshot and git_repo_spec is None:
				git_repo_spec = type(model)

		if timestamp_prefix is None:
			subdir_name = timestamp.strftime(self.timestamp_format)
		else:
			subdir_name = f"{timestamp_prefix}_{timestamp.strftime(self.timestamp_format)}"
		models_groupdir = os.path.join(self.models_dir, dir_prefix, group)
		models_subdir = os.path.join(models_groupdir, subdir_name)

		self.ensure_models_dir_exists()
		try:
			os.makedirs(models_subdir, exist_ok=False)
		except FileExistsError:
			subdir_name += timestamp.strftime(self.timestamp_frac_format)
			models_subdir = os.path.join(models_groupdir, subdir_name)
			os.makedirs(models_subdir, exist_ok=False)
		models_subdir_rel = os.path.join(dir_prefix, group, subdir_name)
		print(f"Directory timestamp: {timestamp}")
		print(f"Models subdirectory: {models_subdir_rel}")
		print(f"Full path: {models_subdir}")

		meta_file_path = os.path.join(models_subdir, self.models_subdir_meta)
		with open(meta_file_path, 'w'):
			pass
		meta_file_stamp = timestamp.timestamp()
		os.utime(meta_file_path, times=(meta_file_stamp, meta_file_stamp))

		tee_logger = None
		if create_tee_logger:
			log_file_name = subdir_name + '.log'
			log_file_path = os.path.join(models_subdir, log_file_name)
			tee_logger = ppyutil.stdtee.StdTee(log_file_path, file_line_buffered=ppyutil.interpreter.debugger_attached())
			print(f"Log file: {log_file_name}")

		git_info = None
		if create_git_snapshot and git_repo_spec is not None:
			git_snapshot_name = subdir_name + '.patch'
			git_snapshot_path = os.path.join(models_subdir, git_snapshot_name)
			git_info = self.create_git_snapshot(git_snapshot_path, git_repo_spec, timestamp=timestamp)
			print(f"Git snapshot: {git_snapshot_name}")

		created_sources = None
		if create_source_snapshot and model is not None:
			source_snapshot_basename = subdir_name + '_orig'
			source_snapshot_dir = models_subdir
			created_sources = self.create_source_snapshot(C, model, source_snapshot_dir, source_snapshot_basename, model_save_kwargs=model_save_kwargs)
			for source_snapshot_profile, source_snapshot_data in created_sources.items():
				print(f"Source snapshot ({source_snapshot_profile}): {source_snapshot_data[1]}")

		print()

		return models_groupdir, models_subdir, subdir_name, timestamp, tee_logger, git_info, created_sources

	def create_git_snapshot(self, git_snapshot_path, git_repo_spec, timestamp=None, force_valid_patch=None):
		# git_snapshot_path = Path to save the git snapshot patch file as
		# git_repo_spec = Object/type that can be used to determine the location of the required git repository
		# timestamp = Optional timestamp to use instead of just taking the current datetime
		# force_valid_patch = Whether to force the file to be of valid patch format by adding a fake empty diff hunk if there are no actual working changes (None => Default)
		# Return a dict of information about the git snapshot

		if timestamp is None:
			timestamp = datetime.datetime.now()
		if force_valid_patch is None:
			force_valid_patch = self.git_force_valid_patch

		git_repo = ppyutil.git.get_git_repo(obj=git_repo_spec)
		git_head = ppyutil.git.get_repo_head(git_repo)
		git_diff = ppyutil.git.all_working_changes(git_repo, tracked_binary=self.git_snapshot_tracked_binary, untracked_binary=self.git_snapshot_untracked_binary)

		git_info = {
			'repo': git_repo.working_dir,
			'stamp': timestamp,
			'commit': git_head.hexsha,
			'commit8': git_head.hexsha[:8],
			'commit_desc': git_head.summary,
			'working_changes': git_diff.count('\n') if git_diff else 0,
			'tracked_binary': self.git_snapshot_tracked_binary,
			'untracked_binary': self.git_snapshot_untracked_binary,
		}

		with open(git_snapshot_path, 'w') as file:
			print(
				f"Git repo: {git_info['repo']}\n"
				f"Timestamp: {git_info['stamp']}\n"
				f"Latest commit: {git_info['commit']}\n"
				f"Commit summary: {git_info['commit_desc']}\n"
				f"Includes tracked binary files: {self.git_snapshot_tracked_binary}\n"
				f"Includes untracked binary files: {self.git_snapshot_untracked_binary}\n"
				f"Working changes: {'See patch below' if git_diff else 'None'}",
				file=file
			)
			if git_diff:
				print(f"\n{git_diff}", file=file)
			elif force_valid_patch:
				print("\n--- empty_diff\n+++ empty_diff\n@@ -1,1 +1,1 @@ empty_diff\n-\n+", file=file)

		return git_info

	def create_source_snapshot(self, C, model, source_snapshot_dir, source_snapshot_basename, reference_snapshot=None, warn_on_overwrite=True, model_save_kwargs=None):
		# C = Configuration of type config.Config
		# model = Model object to construct a source snapshot for
		# source_snapshot_dir = Directory in which to create the source snapshot
		# source_snapshot_basename = Base name to use for the created source files (without *.py file extension)
		# reference_snapshot = A previously generated source snapshot (output of this function) that does not (per-profile) need to be duplicated (if the new source code is identical to the reference snapshot for a profile, then the new source file for that profile is not created)
		# warn_on_overwrite = Warn if any saved file overwrites a previously existing one
		# model_save_kwargs = Keyword arguments to pass to internal calls to get_model_save_data()
		# Return Dict[Profile] = Tuple(file_path, file_name, source_code) of the created source files, where profile corresponds to the profiles defined in get_model_save_data()

		if model_save_kwargs is None:
			model_save_kwargs = {}
		model_data = self.get_model_save_data(C, model, **model_save_kwargs)

		created_sources = {}
		for source_snapshot_profile, model_save_data in model_data.items():

			source_snapshot_name = f"{source_snapshot_basename}_{source_snapshot_profile}.py"
			source_snapshot_path = os.path.join(source_snapshot_dir, source_snapshot_name)
			source_code = ppyutil.pickle.get_pickle_types_source_code(model_save_data, include_builtins=self.source_snapshot_builtins, include_in_prefix=self.source_snapshot_in_prefix)

			if reference_snapshot is not None and source_snapshot_profile in reference_snapshot:
				if source_code == reference_snapshot[source_snapshot_profile][2]:
					if warn_on_overwrite and os.path.exists(source_snapshot_path):
						print_warn(f"Would have created and overwritten the existing source snapshot file {source_snapshot_name}, but didn't as the new data would have duplicated the reference snapshot")
					continue

			if warn_on_overwrite and os.path.exists(source_snapshot_path):
				print_warn(f"Source snapshot file {source_snapshot_name} already exists in the save directory => Overwriting...")
			with open(source_snapshot_path, 'w') as file:
				print(source_code, file=file, end='')

			created_sources[source_snapshot_profile] = (source_snapshot_path, source_snapshot_name, source_code)

		return created_sources

	def save_model_data(self, C, model, meta, save_dir, file_basename=None, add_timestamp=True, timestamp=None, file_ext=SAVED_MODEL_EXT, create_source_snapshot=True, reference_snapshot=None, warn_on_overwrite=True, model_save_kwargs=None):
		# C = Configuration of type config.Config
		# model = Model object to save the data of (one file per profile)
		# meta = Metadata dict to store alongside the model (None => Do not write metadata file)
		# save_dir = Directory to save the model data into
		# file_basename = Base filename to use, NOT including the file extension (None = Auto-generate based on model)
		# add_timestamp = Whether to add a timestamp to the base filename
		# timestamp = Timestamp to use, if provided, otherwise use the current datetime
		# file_ext = File extension to use (with the leading dot)
		# create_source_snapshot = Whether to create an adjoining source snapshot
		# reference_snapshot = Previous source snapshot to compare against, and if the current generated snapshot is identical (per-profile), don't create it
		# warn_on_overwrite = Warn if any saved file overwrites a previously existing one
		# model_save_kwargs = Keyword arguments to pass to internal calls to get_model_save_data()
		# Return Dict[Profile] = Tuple(path, filename) of the created model files, Dict[Profile] = Tuple(file_path, file_name, source_code) of the created source files, Tuple(file_path, file_name, meta) of the created meta file or None

		if model_save_kwargs is None:
			model_save_kwargs = {}

		if file_basename is None:
			file_basename = model.__class__.__name__

		if add_timestamp:
			if timestamp is None:
				timestamp = datetime.datetime.now()
			if file_basename:
				file_basename += f"_"
			file_basename += timestamp.strftime(self.timestamp_format)

		def get_model_path(basename, save_profile):
			filename = f"{basename}_{save_profile}{file_ext}"
			return os.path.join(save_dir, filename), filename

		model_data = self.get_model_save_data(C, model, **model_save_kwargs)

		for profile in model_data:
			model_path, model_file = get_model_path(file_basename, profile)
			if add_timestamp and os.path.exists(model_path):
				print_warn(f"Some model files with the basename {file_basename} already exist in the save directory => Incorporating fractional seconds into basename...")
				file_basename += timestamp.strftime(self.timestamp_frac_format)
				break

		created_model_files = {}
		for profile, data in model_data.items():
			model_path, model_file = get_model_path(file_basename, profile)
			if warn_on_overwrite and os.path.exists(model_path):
				print_warn(f"Model file {model_file} already exists in the save directory => Overwriting...")
			torch.save(data, model_path)
			created_model_files[profile] = (model_path, model_file)

		created_sources = {}
		if create_source_snapshot:
			created_sources = self.create_source_snapshot(C, model, save_dir, file_basename, reference_snapshot=reference_snapshot, warn_on_overwrite=warn_on_overwrite, model_save_kwargs=model_save_kwargs)

		created_meta_file = None
		if isinstance(meta, dict):
			meta_file = file_basename + SAVED_MODEL_META_SUFFIX_EXT
			meta_path = os.path.join(save_dir, meta_file)
			self.save_model_meta(meta_path, meta)
			created_meta_file = (meta_path, meta_file, meta)
		elif meta is not None:
			print_warn(f"Model meta to save should be a dict: {type(meta)}")

		return created_model_files, created_sources, created_meta_file

	@classmethod
	def save_model_meta(cls, meta_path, meta):
		# meta_path = Output meta YAML file path
		# meta = Model metadata to save
		with open(meta_path, 'w') as file:
			yaml.dump(meta, stream=file, indent=2, sort_keys=True, allow_unicode=True, default_flow_style=False, width=2147483647, Dumper=yaml.CSafeDumper)

	@classmethod
	def delete_saved_model_files(cls, created_model_files, created_sources, created_meta_file):
		# created_model_files, created_sources, created_meta_file = See outputs of save_model_data() function
		for file_info in itertools.chain(created_model_files.values(), created_sources.values(), (created_meta_file,)):
			if isinstance(file_info, tuple):
				try:
					os.remove(file_info[0])
				except OSError as e:
					print_warn(f"Removing file {file_info[0]} failed with {e.__class__.__name__}: {e}")
			else:
				print_warn(f"Unable to remove file as the corresponding file info object is not a tuple: {file_info}")

	@classmethod
	def create_log_csv(cls, save_dir, file_basename, template_data):
		# save_dir = Directory to save the CSV log into
		# file_basename = Base filename to use (--> SAVEDIR/BASENAME_log.csv)
		# template_data = Dict[str, Any] of required data columns
		# Return the newly created CSV log path, headers tuple, and name
		csv_name = file_basename + '_log.csv'
		csv_path = os.path.join(save_dir, csv_name)
		csv_headers = tuple(template_data)
		csv_header_io = io.StringIO()
		csv.writer(csv_header_io, dialect='unix', quoting=csv.QUOTE_ALL).writerow(csv_headers)
		with open(csv_path, 'w') as file:
			file.write(csv_header_io.getvalue())
		return csv_path, csv_headers, csv_name

	@classmethod
	def write_log_csv(cls, csv_path, csv_headers, data):
		# csv_path = CSV log path returned from create_log_csv()
		# csv_headers = CSV log headers tuple returned from create_log_csv()
		# data = Data to write to the CSV log
		# Return whether the data was as expected (no change in columns or order)
		csv_row_io = io.StringIO()
		csv.writer(csv_row_io, dialect='unix', quoting=csv.QUOTE_MINIMAL).writerow(f'{value:#.4g}' if isinstance(value, float) else str(value) for value in data.values())
		with open(csv_path, 'a') as file:
			file.write(csv_row_io.getvalue())
		data_headers = tuple(data)
		return data_headers == csv_headers

	@classmethod
	def save_results_data(cls, R, save_dir, file_basename, save_pickle=True, save_yaml=True, save_json=True, always_simplify=False):
		# R = Results data to save to disk
		# save_dir = Directory to save the results data into
		# file_basename = Base filename to use (--> SAVEDIR/BASENAME_results.EXT)
		# save_pickle = Whether to save the data in pickle format
		# save_yaml = Whether to save the data in YAML format
		# save_json = Whether to save the data in JSON format
		# always_simplify = Whether to force the generation of the simplified output data no matter which output formats are actually enabled
		# Return the simplified data that was saved to the YAML/JSON formats (None if both formats were disabled)

		print("Saving results data:")
		basepath = os.path.join(save_dir, file_basename + '_results')

		Rsimple = None
		saved = False

		if save_pickle:
			results_pickle = basepath + '.pkl'
			with open(results_pickle, 'wb') as file:
				pickle.dump(R, file)
			print(f"  Saved pickle: {results_pickle}")
			saved = True

		if save_yaml or save_json or always_simplify:
			Rsimple = ppyutil.objmanip.simplify_object(R)

		if save_yaml:
			results_yaml = basepath + '.yaml'
			with open(results_yaml, 'w') as file:
				yaml.dump(Rsimple, stream=file, indent=2, sort_keys=True, allow_unicode=True, default_flow_style=False, width=2147483647, Dumper=yaml.CSafeDumper)
			print(f"  Saved YAML: {results_yaml}")
			saved = True

		if save_json:
			results_json = basepath + '.json'
			with open(results_json, 'w') as file:
				json.dump(Rsimple, file, indent=2, sort_keys=True, ensure_ascii=False)
			print(f"  Saved JSON: {results_json}")
			saved = True

		if not saved:
			print("  All target formats disabled => Nothing was saved!")
		print()

		return Rsimple

	@classmethod
	def save_results_csv(cls, R, save_dir, file_basename, CSVF):
		# R = Results data to save in CSV form (ideally this should be a simplified object, see ppyutil.objmanip.simplify_object())
		# save_dir = Directory to save the CSV into
		# file_basename = Base filename to use (--> SAVEDIR/BASENAME_results.csv)
		# CSVF = yaml_spec.YAMLSpec of format (name, spec), where spec is a dict specifying the required output format of the CSV (fields: columns = List(str), names = Dict[str] -> str, all = Bool)
		# Return the flattened version of R that was used to save the CSV

		print("Saving results data to CSV:")
		print(f"  CSV format: {CSVF.name}")
		Rflat = ppyutil.objmanip.flatten_object_dicts(R, flatten_all=True)
		print(f"  Number of unique columns available = {len(Rflat)}")

		csv_fmt = CSVF.spec

		csv_all = csv_fmt.get('all', True)
		if not isinstance(csv_all, bool):
			print_warn(f"CSV format specifier 'all' should be boolean: {csv_all}")
			csv_all = bool(csv_all)

		csv_columns = csv_fmt.get('columns', [])
		if isinstance(csv_columns, list):
			csv_columns = csv_columns.copy()
		else:
			print_warn(f"CSV format specifier 'columns' should be a list (got {type(csv_columns)}) => Forcing CSV output of all available data")
			csv_columns = []
			csv_all = True
		csv_columns[:] = [str(c) for c in csv_columns]

		csv_groups = csv_fmt.get('column_groups', {})
		if isinstance(csv_groups, dict):
			csv_groups = {str(group): [str(v) for v in value] for group, value in csv_groups.items() if isinstance(value, list)}
			seen_groups = set()
			i = 0
			while i < len(csv_columns):
				col = csv_columns[i]
				if col in csv_groups:
					if col in seen_groups:
						print_warn(f"Found cycle while expanding the column group {col} => Ignoring the nested copy of this column group")
						del csv_columns[i]
					else:
						csv_columns[i:i+1] = csv_groups[col]
						seen_groups.add(col)
				else:
					i += 1
		else:
			print_warn(f"CSV format specifier 'column_groups' should be a dict of lists (got {type(csv_groups)}) => Ignoring")

		print(f"  Number of explicitly selected data columns = {len(csv_columns)} (not necessarily unique)")

		csv_names = csv_fmt.get('names', {})
		if not isinstance(csv_names, dict):
			print_warn(f"CSV format specifier 'names' should be a dict (got {type(csv_names)}) => Ignoring")
			csv_names = {}

		csv_basepath = os.path.join(save_dir, f'{file_basename}_results_{CSVF.name}')

		def write_results_to_csv(columns, names, includes_all):

			names = {str(key): str(mapped) for key, mapped in names.items() if str(key) in columns}

			csv_id = base64.b64encode(hashlib.md5(json.dumps(columns, sort_keys=True, ensure_ascii=False).encode('utf-8')).digest(), altchars=b'#$')[:8].decode('utf-8')
			print(f"  CSV format hash ID = {csv_id}")
			if includes_all:
				print("    Saving all available data in addition to explicitly selected data columns")
			else:
				print("    Saving only explicitly selected data columns")
			print(f"    Number of relevant name remappings = {len(names)}")
			print(f"    Total number of output CSV columns = {len(columns)}")

			if len(columns) <= 0:
				print_warn("No columns selected for outputting to CSV => Nothing to do")
				return

			csv_header_io = io.StringIO()
			csv_named_columns = [names[c] if c in names else re.sub(r'[/_]', r' ', c) for c in columns]
			csv.writer(csv_header_io, dialect='unix', quoting=csv.QUOTE_ALL).writerow(csv_named_columns)
			csv_header = csv_header_io.getvalue()

			results_csv = f"{csv_basepath}{'_all' if includes_all else ''}_{csv_id}.csv"
			print(f"    Output CSV file: {results_csv}")

			csv_row_data = []
			for c in columns:
				if c and c in Rflat:
					value = Rflat[c]
					if isinstance(value, float):
						csv_row_data.append(f'{value:#.4g}')
					else:
						csv_row_data.append(str(value))
				else:
					csv_row_data.append('')

			try:
				with portalocker.Lock(results_csv, mode='a+', timeout=8, newline='') as file:
					if file.tell() <= 0:
						print("    CSV does not exist yet => Creating new one with single results row")
						file.write(csv_header)
					else:
						file.seek(0)
						actual_header = file.readline()
						file.seek(0, 2)
						if actual_header.rstrip('\n\r') != csv_header.rstrip('\n\r'):
							actual_num_cols = len(next(csv.reader(io.StringIO(actual_header), dialect='unix', quoting=csv.QUOTE_ALL)))
							if len(columns) == actual_num_cols:
								print_warn("CSV header does not match our expectations (despite a hash match), but has the right number of columns => Were some columns just renamed?")
							else:
								print_warn("CSV header does not match the required format specification => Outputting repeated header line")
								file.write(csv_header)
						print("    CSV exists => Appending results row")
					csv.writer(file, dialect='unix', quoting=csv.QUOTE_MINIMAL).writerow(csv_row_data)
				print("    CSV data saved successfully")
			except portalocker.exceptions.LockException:
				print_warn(f"Timed out while attempting to acquire lock on file: {results_csv}")
				print("    Failed to save CSV data")

		write_results_to_csv(csv_columns, csv_names, False)
		if csv_all:
			if csv_columns:
				csv_columns.append('')
			csv_columns.extend(str(k) for k in Rflat.keys())
			write_results_to_csv(csv_columns, csv_names, True)
		print()

		return Rflat

	#
	# Action implementations
	#

	def model_info_impl(self, C, load_model_opts=None, load_dataset_opts=None):
		# C = Configuration of type config.Config
		# load_model_opts = Custom options for loading the model
		# load_dataset_opts = Custom options for loading the dataset

		self.apply_global_config(C, allow_cudnn_bench=False)
		model, data_loaders = self.load_network(C, load_model_opts=load_model_opts, load_dataset_opts=load_dataset_opts)[:2]
		model.eval()

		def test_batch(title, data_loader):
			print(f"{title}:")
			data_loader_iter = iter(data_loader)
			data, target = next(data_loader_iter)
			if len(data) != 1:
				raise ValueError(f"Model info is only supported for models with a single input (have {len(data)})")
			with torch.inference_mode():
				data = tuple(d.to(model.device, non_blocking=data_loader.pin_memory) for d in data)
				print(f"  Model input:  {tensor_util.brief_repr(data)}")
				target = tuple(target)
				print(f"  Model target: {tensor_util.brief_repr(target)}")
				output = model(data)
			print(f"  Model output: {tensor_util.brief_repr(output)}")
			print()

		test_batch("Training batch", data_loaders.train)
		test_batch("Validation batch", data_loaders.valid)
		test_batch("Test batch", data_loaders.test)

		if tensorwatch:
			print("Model statistics:")
			test_loader_iter = iter(data_loaders.test)
			test_data, test_target = next(test_loader_iter)
			input_size = list(test_data[0].shape)
			input_size[0] = 1
			model.cpu()
			with torch.inference_mode():
				# noinspection PyUnresolvedReferences
				stats = tensorwatch.model_stats(model, input_size)
			print()
			print(stats)
			print()

	def draw_model_impl(self, C, pdfpath=None, load_model_opts=None, load_dataset_opts=None):
		# C = Configuration of type config.Config
		# pdfpath = String path of the required output pdf file (None => Just return the generated graph)
		# load_model_opts = Custom options for loading the model
		# load_dataset_opts = Custom options for loading the dataset

		if not tensorwatch:
			print_warn("Cannot draw model without the tensorwatch library => Check it's installed and available?")
			print()
			return None

		self.apply_global_config(C, allow_cudnn_bench=False)
		model, data_loaders = self.load_network(C, load_model_opts=load_model_opts, load_dataset_opts=load_dataset_opts)[:2]
		model.eval()

		print("Draw model:")
		test_loader_iter = iter(data_loaders.test)
		test_data, test_target = next(test_loader_iter)
		if len(test_data) != 1:
			raise ValueError(f"Draw model is only supported for models with a single input (have {len(test_data)})")
		input_size = list(test_data[0].shape)
		input_size[0] = 1
		with torch.inference_mode():
			test_data = tuple(d.to(model.device, non_blocking=data_loaders.test.pin_memory) for d in test_data)
			print(f"  Model input:   {tensor_util.brief_repr(test_data)}")
			test_output = model(test_data)
			print(f"  Model output:  {tensor_util.brief_repr(test_output)}")

		model.cpu()
		print("  Creating model graph...")
		with torch.inference_mode():
			# noinspection PyUnresolvedReferences
			drawing = tensorwatch.draw_model(model, input_size)  # TODO: This was broken by a torch update => Maybe torch or tensorwatch fixes it at some point?
		if pdfpath is not None:
			pdfpath = os.path.abspath(pdfpath)
			drawing.save(pdfpath, format='pdf')
			print(f"  Saved graph to pdf: {pdfpath}")
		print()

		return drawing

	def dataset_stats_impl(self, C, basis=DatasetStatsBasis.Default, quick=False, load_model_opts=None, load_dataset_opts=None):
		# C = Configuration of type config.Config
		# basis = Dataset statistics basis of type DatasetStatsBasis
		# quick = Whether to avoid long running operations
		# load_model_opts = Custom options for loading the model
		# load_dataset_opts = Custom options for loading the dataset

		self.apply_global_config(C, allow_cudnn_bench=False)
		model, data_loaders, datasets = self.load_network(C, load_model_opts=load_model_opts, load_dataset_opts=load_dataset_opts)[:3]
		model.eval()

		if quick:
			return

		num_groups = len(model.reqd_inputs.groups)
		if num_groups <= 0:
			raise ValueError("There should be at least one group in the required inputs")

		channel_types_str = tuple(' + '.join(model.reqd_inputs.channels[ch].type.name for ch in model.reqd_inputs.groups[g]) for g in range(num_groups))
		if any(not s for s in channel_types_str):
			raise ValueError("Every group in the required inputs should have at least one channel spec")
		num_channels = None

		@dataclasses.dataclass
		class DatasetStats:
			min: float = math.inf
			max: float = -math.inf
			count: int = 0
			mean: float = 0
			M2: float = 0

		total_samples = 0
		total_dimensions = [collections.Counter() for _ in range(num_groups)]
		total_channel_stats = None

		with torch.inference_mode():
			for dt, data_loader in data_loaders.items():

				dt_caps = dt.capitalize()
				print(f"{dt_caps} dataset:")

				num_samples = 0
				batch_size_counter = collections.Counter()
				dimension_counters = [collections.Counter() for _ in range(num_groups)]
				channel_stats = None

				for data, target in data_loader:

					if len(data) != num_groups:
						raise ValueError(f"{dt_caps} dataset has a batch with {len(data)} groups although {num_groups} is expected")

					batch_size = data[0].shape[0]
					batch_size_counter[batch_size] += 1
					if any(d.shape[0] != batch_size for d in data):
						raise ValueError(f"{dt_caps} dataset has a batch with inconsistent batch sizes amongst the groups (not all {batch_size})")
					num_samples += batch_size

					if num_channels is None:
						num_channels = [d.shape[1] for d in data]
					elif any(data[g].shape[1] != num_channels[g] for g in range(num_groups)):
						raise ValueError(f"{dt_caps} dataset has a batch with unexpected channel sizes: {[data[g].shape[1] for g in range(num_groups)]} vs {num_channels}")

					data = tuple(d.to(model.device, non_blocking=data_loader.pin_memory) for d in data)

					if total_channel_stats is None:
						total_channel_stats = [[DatasetStats() for _ in range(num_channels[g])] for g in range(num_groups)]
					if channel_stats is None:
						channel_stats = [[DatasetStats() for _ in range(num_channels[g])] for g in range(num_groups)]

					for g in range(num_groups):
						gstats = channel_stats[g]
						gvalue = data[g]

						payload_dimension = tuple(gvalue.shape[:1:-1])
						dimension_counters[g][payload_dimension] += batch_size

						reduce_dims = tuple(index for index in range(gvalue.dim()) if index != 1)
						batch_min = gvalue.amin(reduce_dims).tolist()
						batch_max = gvalue.amax(reduce_dims).tolist()
						batch_var, batch_mean = torch.var_mean(gvalue, reduce_dims, unbiased=False)
						batch_mean = batch_mean.tolist()
						batch_var = batch_var.tolist()

						if basis == DatasetStatsBasis.Element:
							count = batch_size * math.prod(payload_dimension)
						elif basis == DatasetStatsBasis.Sample:
							count = batch_size
						elif basis == DatasetStatsBasis.Batch:
							count = 1
						else:
							raise ValueError(f"Unrecognised dataset statistics basis: {basis}")

						for c in range(num_channels[g]):
							dstats = gstats[c]
							bmin = batch_min[c]
							bmax = batch_max[c]
							bmean = batch_mean[c]
							bvar = batch_var[c]

							if bmin < dstats.min:
								dstats.min = bmin
							if bmax > dstats.max:
								dstats.max = bmax

							old_count = dstats.count
							dstats.count += count
							delta = bmean - dstats.mean
							delta_mean = delta * (count / dstats.count)
							dstats.mean += delta_mean
							dstats.M2 += count * bvar + old_count * (delta * delta_mean)

				if num_channels is None:
					raise ValueError(f"Failed to deduce number of channels in each group => Is there even a single sample in the {dt} dataset?")
				elif channel_stats is None:
					raise ValueError(f"Failed to calculate channel statistics => Is there even a single sample in the {dt} dataset?")

				print(f"  Num samples: {num_samples}")
				print(f"  Num batches: {len(data_loader)}")
				print("  Batch sizes: " + ', '.join(f"{size} (\xD7{count})" for size, count in batch_size_counter.most_common()))
				for g in range(num_groups):
					print(f"  Group {g}:")
					print(f"    Channel types: {channel_types_str[g]}")
					print(f"    Num channels:  {num_channels[g]}")
					print("    Dimensions: " + ', '.join('\xD7'.join(str(s) for s in size) + f" (\xD7{count})" for size, count in dimension_counters[g].most_common()))
					gstats = channel_stats[g]
					for c in range(num_channels[g]):
						dstats = gstats[c]
						print(f"    Channel {c} ({basis.name.lower()}): {dstats.min:.6g} \u2264 {dstats.mean:.6g} \xB1 {math.sqrt(dstats.M2 / dstats.count):.6g}\u03BB \u2264 {dstats.max:.6g}")
				print()

				total_samples += num_samples
				for g in range(num_groups):
					total_dimensions[g].update(dimension_counters[g])
					tgstats = total_channel_stats[g]
					gstats = channel_stats[g]
					for c in range(num_channels[g]):
						tdstats = tgstats[c]
						dstats = gstats[c]

						if dstats.min < tdstats.min:
							tdstats.min = dstats.min
						if dstats.max > tdstats.max:
							tdstats.max = dstats.max

						old_count = tdstats.count
						tdstats.count += dstats.count
						tdstats.mean = (old_count * tdstats.mean + dstats.count * dstats.mean) / tdstats.count
						delta = dstats.mean - tdstats.mean
						tdstats.M2 += dstats.M2 + delta * delta * (old_count * dstats.count / tdstats.count)

		print("Total dataset:")
		print(f"  Num samples: {total_samples}")
		for g in range(num_groups):
			print(f"  Group {g}:")
			print(f"    Channel types: {channel_types_str[g]}")
			print(f"    Num channels:  {num_channels[g]}")
			print("    Dimensions: " + ', '.join('\xD7'.join(str(s) for s in size) + f" (\xD7{count})" for size, count in total_dimensions[g].most_common()))
			gstats = total_channel_stats[g]
			for c in range(num_channels[g]):
				dstats = gstats[c]
				print(f"    Channel {c} ({basis.name.lower()}): {dstats.min:.6g} \u2264 {dstats.mean:.6g} \xB1 {math.sqrt(dstats.M2 / dstats.count):.6g}\u03BB \u2264 {dstats.max:.6g}")
		print()

	def train_network_impl(self, C, CSVF=None, group=None, epochs=None, load_model_opts=None, load_dataset_opts=None):
		# C = Configuration of type config.Config (see also resolve_configuration() function)
		# CSVF = CSV format of type yaml_spec.YAMLSpec (see also resolve_csvfmt() function)
		# group = Group to assign this training run to (None => Assign to default group)
		# epochs = Maximum number of epochs to train (overrides configuration if >= 1)
		# load_model_opts = Custom options for loading the model
		# load_dataset_opts = Custom options for loading the dataset
		# Return a dict with data on the training environment and training results

		string_tee = ppyutil.stdtee.StdTeeString()
		with string_tee:

			C = self.resolve_configuration(C)
			CSVF = self.resolve_csvfmt(CSVF)

			# noinspection PyDictCreation
			R: Dict[str, Any] = {'C': C, 'Configuration': C.name(), 'CSVFmt': CSVF.name}

			sys_info = self.apply_global_config(C)
			R['SysInfo'] = sys_info

			action_cm = self.ensure_entered(Context.Action)

			if not group:
				group = self.default_group
			R['Group'] = group
			print(f"Training run is assigned to group: {group}")
			print(f"Results CSV format: {CSVF.name}")
			print()

			model, data_loaders, datasets, model_info, dataset_info, model_lock = self.load_network(C, load_model_opts=load_model_opts, load_dataset_opts=load_dataset_opts)[:6]
			R['Model'] = model_info
			R['Dataset'] = dataset_info

			nvsmi = ppyutil.nvsmi.NvidiaSMI()
			training.reset_gpu_memory_status_peaks(model.device)

			def get_model_dims(obj):
				str_list = []
				for o in obj:
					if o is None:
						str_list.append(str(o))
					elif torch.is_tensor(o):
						osize = o.size()
						str_list.append('x'.join(str(d) for d in reversed(osize[1:])))
					else:
						str_list.append('?')
				return '|'.join(str_list)

			RModel = R['Model']
			RModel['input_dims'] = None   # Note: This may remain None if no single batch is trained
			RModel['output_dims'] = None  # Note: This may remain None if no single batch is trained
			if C.Deterministic:
				model_hash_init = model_util.model_param_hash(model)
				print(f"Initial model parameter MD5 hash: {model_hash_init}")
				print()
			else:
				model_hash_init = None
			RModel['md5_init'] = model_hash_init  # Note: This is None if we are not training deterministically

			criterion = self.load_criterion(model)
			optimizer = self.load_optimizer(model)
			scheduler, reqd_scheduler_args = self.load_scheduler(C, optimizer, model)
			stopper = self.load_stopper(C)

			models_groupdir, models_subdir, subdir_name, timestamp, tee_logger, git_info, source_snapshot = self.create_models_subdir('train_network', group, C=C, model=model)
			R['Run'] = {'name': subdir_name, 'group_dir': models_groupdir, 'dir': models_subdir, 'stamp': timestamp}
			R['Git'] = git_info  # Note: This may be None if no git snapshot was made

		with tee_logger as log:
			log.write(string_tee.value())

			if model.device.type == 'cuda':
				training.show_nvidia_smi_status(model.device, nvsmi)
				training.show_gpu_memory_status(model.device)
				print()

			ref_losses, min_ref_losses, ref_time = training.eval_reference_losses(model.device, model, criterion, data_loaders, enabled=C.TrainRefLosses)
			ref_time_avg = ref_time / len(ref_losses) if ref_losses else 0
			R['RefLoss'] = {
				'min': min_ref_losses,
				'all': {name: losses for name, losses in ref_losses},
				'time': ref_time,
				'time_avg': ref_time_avg,
			}

			if epochs is not None and epochs >= 1:
				num_epochs = epochs
			else:
				num_epochs = max(C.TrainEpochs, 1)
			print(f"Training up to {num_epochs} epochs:")

			min_train_loss = math.inf   # Lowest seen training loss
			min_valid_loss = math.inf   # Lowest seen validation loss
			min_test_loss = math.inf    # Lowest seen test loss

			best_train_loss = math.inf  # Training loss at the best epoch
			best_valid_loss = math.inf  # Validation loss at the best epoch
			best_test_loss = math.inf   # Test loss at the best epoch
			best_epoch = 1              # Number of the best epoch

			new_best_epochs = []
			scheduler_events = []
			model_saves = collections.deque()
			perf_start_epoch = 3
			perf_start_time = None
			smi_infos = []
			gpu_mem_statuses = []

			tprint = ppyutil.print.TimedPrinter()
			tprint_lock = action_cm.enter_context(self.run_lock.config(newline=False, block_newline=False, file=tprint))

			if C.TrainLogCSV:
				log_csv_data = {
					'Epoch': 0,
					'Rel train loss': 0,
					'Rel valid loss': 0,
					'Rel test loss': 0,
					'Learning rate': '',
					'SMI MiB': 0,
					'Saved model': '',
				}
				log_csv_path, log_csv_headers, log_csv_name = self.create_log_csv(models_subdir, subdir_name, log_csv_data)
				tprint(f"Creating training log: {log_csv_name}")
			else:
				log_csv_data = None
				log_csv_path = None
				log_csv_headers = None

			initial_lr = [group['lr'] for group in optimizer.param_groups]
			tprint(f"Initial learning rate{'s' if len(initial_lr) > 1 else ''}: {training.lr_string(initial_lr, brackets=False)}")
			best_lr = initial_lr

			smi_info = training.show_nvidia_smi_status(model.device, nvsmi, file=tprint)
			gpu_temp_lowpass = ppyutil.filter.LowPassFilter(settling_time=C.GPUTempLowPassTs, init_value=smi_info['temp'] if smi_info else 40)
			training.show_gpu_memory_status(model.device, file=tprint)

			solo_lock = None
			high_exec_lock = action_cm.enter_context(self.run_lock.level(GPULevel.NormalMemHighExec))
			if C.TrainLockFirstEpochs != 0:
				solo_lock = action_cm.enter_context(self.run_lock.solo(ensure_level=GPULevel.HighMemHighExec))

			actual_start_time = tprint.current_time()

			def epoch_separator():
				tprint('-' * 80)

			for epoch in range(1, num_epochs + 1):

				if epoch == perf_start_epoch:
					perf_start_time = tprint.current_time()

				gpu_mem_statuses.append(training.reset_gpu_memory_status_peaks(model.device, query=True))

				epoch_separator()
				tprint(f"START EPOCH {epoch}")

				current_lr = [group['lr'] for group in optimizer.param_groups]
				current_lr_string = training.lr_string(current_lr, brackets=False)
				tprint(f"Current learning rate{'s' if len(current_lr) > 1 else ''}: {current_lr_string}")

				model.train()

				train_loss = 0.0
				batch_avg_loss_max = -math.inf
				batch_avg_loss_min = math.inf
				num_trained = 0
				num_train_batches = len(data_loaders.train)

				for data, target in data_loaders.train:

					num_in_batch = data[0].shape[0]
					incomplete_batch = num_in_batch < data_loaders.train.batch_size

					data = tuple(d.to(model.device, non_blocking=data_loaders.train.pin_memory) for d in data)
					target = tuple(t.to(model.device, non_blocking=data_loaders.train.pin_memory) for t in target)

					if RModel['input_dims'] is None:
						RModel['input_dims'] = get_model_dims(data)

					optimizer.zero_grad()
					output = model(data)
					batch_avg_loss = criterion(output, target)
					batch_avg_loss.backward()
					optimizer.step()

					if RModel['output_dims'] is None:
						RModel['output_dims'] = get_model_dims(output)

					batch_avg_loss_float = batch_avg_loss.item()
					if not incomplete_batch:
						batch_avg_loss_min = min(batch_avg_loss_min, batch_avg_loss_float)
						batch_avg_loss_max = max(batch_avg_loss_max, batch_avg_loss_float)

					train_loss += batch_avg_loss_float * num_in_batch
					num_trained += num_in_batch

				train_loss /= num_trained
				min_train_loss = min(min_train_loss, train_loss)

				smi_info = training.show_nvidia_smi_status(model.device, nvsmi, file=tprint)
				smi_infos.append(smi_info)
				training.show_gpu_memory_status(model.device, file=tprint)

				if smi_info:
					filtered_gpu_temp = gpu_temp_lowpass.filter(smi_info['temp'])
					filtered_gpu_temp_thres = smi_info['temp_slow'] - C.GPUTempSafetyMargin
					if filtered_gpu_temp > filtered_gpu_temp_thres:
						mail_to = None
						try:
							device_index = device_util.resolve_device_index(model.device, enforce_type='cuda')
							gpu_info_list = sys_info['gpu']['list']
							if device_index < len(gpu_info_list):
								device_str = f"GPU {device_index} ({gpu_info_list[device_index]['pci_id']})"
								device_model = gpu_info_list[device_index]['name']
							else:
								device_str = str(model.device)
								device_model = "Unknown"
							network_name = platform.node()
							mail_to = f"{pwd.getpwuid(os.getuid()).pw_name}@{network_name}"
							mail_msg = (
								"GPU overheating was detected, so CUDA training was aborted for safety reasons!\n\n"
								"DETAILS:\n"
								f"Computer = {network_name}\n"
								f"GPU device = {device_str}\n"
								f"GPU model = {device_model}\n"
								f"Timestamp = {datetime.datetime.now()}\n"
								f"Log file = {tee_logger.file_path}\n"
								f"Current GPU temp = {smi_info['temp']}{smi_info['temp_unit']}\n"
								f"Low-pass GPU temp = {filtered_gpu_temp:.1f}{smi_info['temp_unit']}\n"
								f"Overheating threshold = {filtered_gpu_temp_thres:.1f}{smi_info['temp_unit']}\n"
							)
							subprocess.run(["/usr/bin/mail", "-s", "GPU overheating error", mail_to], input=mail_msg.encode('utf-8'))
						finally:
							raise GPUHardwareError(f"GPU overheating detected (current {smi_info['temp']}{smi_info['temp_unit']}, filtered {filtered_gpu_temp:.1f}{smi_info['temp_unit']} > {filtered_gpu_temp_thres:.1f}{smi_info['temp_unit']}) => Aborting process and sending notification mail to {mail_to}!")

				tprint(f"Trained {num_trained} samples in {num_train_batches} batches: Mean loss {train_loss:{loss_fmt}} (min {min_train_loss:{loss_fmt}}, range {batch_avg_loss_min:{loss_fmt}} -> {batch_avg_loss_max:{loss_fmt}})")

				test_loss, num_tested, num_test_batches = training.eval_model_loss(model, criterion, data_loaders.test)
				min_test_loss = min(min_test_loss, test_loss)

				tprint(f"Tested {num_tested} samples in {num_test_batches} batches: Mean loss {test_loss:{loss_fmt}} (min {min_test_loss:{loss_fmt}})")

				valid_loss, num_valided, num_valid_batches = training.eval_model_loss(model, criterion, data_loaders.valid)
				min_valid_loss = min(min_valid_loss, valid_loss)

				new_best = False
				created_model_files = None
				if valid_loss <= best_valid_loss:
					if not C.TrainSaveOnlyIfBeatsRef or valid_loss <= min_ref_losses.valid:
						model_meta = {'Epoch': epoch}
						created_files = self.save_model_data(C, model, model_meta, models_subdir, create_source_snapshot=True, reference_snapshot=source_snapshot)
						created_model_files = created_files[0]
						model_saves.append(created_files)
						while len(model_saves) > C.TrainSaveNumLatest >= 1:
							self.delete_saved_model_files(*model_saves.popleft())
					best_train_loss = train_loss
					best_valid_loss = valid_loss
					best_test_loss = test_loss
					best_epoch = epoch
					best_lr = current_lr
					new_best = True
					new_best_epochs.append(epoch)

				new_best_model = None
				if new_best:
					if created_model_files:
						new_best_model = '|'.join(sorted(data[1] for data in created_model_files.values()))
						best_spec = f" [NEW BEST: {new_best_model}]"
					else:
						best_spec = f" [NEW BEST: Not saved, as not better than reference losses]"
				else:
					best_spec = f" (min {min_valid_loss:{loss_fmt}})"
				tprint(f"Validated {num_valided} samples in {num_valid_batches} batches: Mean loss {valid_loss:{loss_fmt}}{best_spec}")

				if epoch == C.TrainLockFirstEpochs:
					action_cm.leave_context(solo_lock)
					solo_lock = None

				if log_csv_path is not None:
					log_csv_data['Epoch'] = epoch
					log_csv_data['Rel train loss'] = train_loss / min_ref_losses.train if ref_losses else train_loss
					log_csv_data['Rel valid loss'] = valid_loss / min_ref_losses.valid if ref_losses else valid_loss
					log_csv_data['Rel test loss'] = test_loss / min_ref_losses.test if ref_losses else test_loss
					log_csv_data['Learning rate'] = current_lr_string
					log_csv_data['SMI MiB'] = smi_info['mem_us'] if smi_info is not None and smi_info['mem_us'] is not None else 0
					log_csv_data['Saved model'] = new_best_model if new_best_model is not None else ''
					if not self.write_log_csv(log_csv_path, log_csv_headers, log_csv_data):
						print_warn(f"Log CSV headers have unexpectedly changed to: {tuple(log_csv_data)}", prefix=False, file=tprint)

				epoch_losses = dataset.DatasetTuple(train=train_loss, valid=valid_loss, test=test_loss)
				best_losses = dataset.DatasetTuple(train=best_train_loss, valid=best_valid_loss, test=best_test_loss)
				min_losses = dataset.DatasetTuple(train=min_train_loss, valid=min_valid_loss, test=min_test_loss)
				should_stop, stopping_reason = stopper.should_stop(epoch, epoch_losses, best_losses, min_losses, initial_lr, current_lr)
				if stopping_reason:
					tprint(f"Stopper: {stopping_reason}")
				if should_stop:
					tprint(f"Early stopping condition triggered")
					break

				if reqd_scheduler_args == LRSchedulerArgs.NoArgs:
					scheduler.step()
				else:
					scheduler_data = {LRSchedulerArgs.ValidLoss: valid_loss, LRSchedulerArgs.MinRefValidLoss: min_ref_losses.valid}
					if isinstance(reqd_scheduler_args, LRSchedulerArgs):
						scheduler.step(scheduler_data[reqd_scheduler_args])
					else:
						scheduler_args = [scheduler_data[arg] for arg in reqd_scheduler_args]
						scheduler.step(*scheduler_args)
				status_string_fn = getattr(scheduler, 'status_string', None)
				if callable(status_string_fn):
					tprint(f"LR scheduler: {status_string_fn()}")
				event_string_fn = getattr(scheduler, 'event_string', None)
				if callable(event_string_fn):
					event_string = event_string_fn()
					if event_string:
						scheduler_events.append((epoch, event_string))

				while self.run_lock.solo_pending:
					self.run_lock.yield_to_solo()

			epoch_separator()
			end_time = tprint.current_time()

			if solo_lock is not None and C.TrainLockFirstEpochs > 0:
				action_cm.leave_context(solo_lock)
			action_cm.leave_context(tprint_lock)

			print(f"FINISHED {epoch} EPOCHS")
			print()

			final_lr = current_lr
			perf_epochs = epoch - perf_start_epoch + 1
			if perf_start_time is None or perf_start_time <= 0 or perf_epochs < 1:
				perf_epochs = epoch
				perf_time = end_time - actual_start_time
			else:
				perf_time = max(end_time - perf_start_time, 1e-6)
			perf_epoch_time = perf_time / perf_epochs
			perf_epochs_per_h = 3600 * perf_epochs / perf_time

			if model.device.type == 'cuda':

				training.show_nvidia_smi_status(model.device, nvsmi)
				gpu_mem_statuses.append(training.show_gpu_memory_status(model.device))
				print()

				R['GPU'] = {}  # Note: The key 'GPU' is only available in R if training with CUDA was used
				RGPU: Dict[str, Any] = R['GPU']

				gpu_mem_statuses = [s for s in gpu_mem_statuses if s]
				if gpu_mem_statuses:
					mem_alloc_peak = [s['alloc']['peak'] for s in gpu_mem_statuses]
					mem_reserved_peak = [s['reserved']['peak'] for s in gpu_mem_statuses]
					RGPU['mem'] = {  # Note: If have_info is False then the remaining fields in 'mem' are not guaranteed to exist
						'have_info': True,
						'alloc_peak_avg': round(statistics.mean(mem_alloc_peak)),
						'alloc_peak_max': max(mem_alloc_peak),
						'reserved_peak_avg': round(statistics.mean(mem_reserved_peak)),
						'reserved_peak_max': max(mem_reserved_peak),
					}
				else:
					RGPU['mem'] = {'have_info': False}

				smi_data = smi_infos[2:] if len(smi_infos) > 2 else smi_infos
				smi_data = [d for d in smi_data if d]
				if smi_data:
					smi_gpu_temps = [d['temp'] for d in smi_data]
					smi_gpu_utils = [d['gpu_util'] for d in smi_data]
					smi_mem_utils = [d['mem_util'] for d in smi_data]
					smi_mem_us = [d['mem_us'] for d in smi_data if d['mem_us'] is not None]
					smi_pstates = [d['pstate'] for d in smi_data if d['pstate'] is not None]
					RGPU['smi'] = {  # Note: If have_info is False then the remaining fields in 'smi' are not guaranteed to exist
						'have_info': True,
						'temp_avg': round(statistics.mean(smi_gpu_temps)),
						'temp_max': max(smi_gpu_temps),
						'temp_slow': smi_data[-1]['temp_slow'],
						'temp_crit': smi_data[-1]['temp_crit'],
						'gpu_util_avg': round(statistics.mean(smi_gpu_utils)),
						'gpu_util_max': max(smi_gpu_utils),
						'mem_util_avg': round(statistics.mean(smi_mem_utils)),
						'mem_util_max': max(smi_mem_utils),
						'mem_us_avg': round(statistics.mean(smi_mem_us)) if smi_mem_us else 0,
						'mem_us_max': max(smi_mem_us, default=0),
						'pstate_min': min(smi_pstates, default=99),
						'pstate_avg': statistics.mean(smi_pstates) if smi_pstates else 99,
						'pstate_max': max(smi_pstates, default=99),
					}
				else:
					RGPU['smi'] = {'have_info': False}

			RRun = R['Run']
			RRun['epochs'] = epoch
			RRun['epoch_limit'] = num_epochs
			RRun['best_epoch'] = best_epoch
			RRun['time'] = end_time - actual_start_time
			RRun['epoch_time'] = perf_epoch_time
			RRun['epochs_per_h'] = perf_epochs_per_h

			R['LR'] = {
				'initial': training.lr_string(initial_lr, brackets=False, sep='|'),
				'at_best': training.lr_string(best_lr, brackets=False, sep='|'),
				'final': training.lr_string(final_lr, brackets=False, sep='|'),
				'initial0': initial_lr[0],
				'at_best0': best_lr[0],
				'final0': final_lr[0],
			}

			if C.Deterministic:
				model_hash_final = model_util.model_param_hash(model)
				print(f"Finals model parameter MD5 hash: {model_hash_final}")
				print()
			else:
				model_hash_final = None
			RModel['md5_final'] = model_hash_final  # Note: This is None if we are not training deterministically

			if scheduler_events:
				print("Learning rate event summary:")
				last_epoch = 0
				for epoch, event in scheduler_events:
					best_epochs = [str(e) for e in new_best_epochs if last_epoch < e <= epoch]
					if best_epochs:
						print(f"  New best epochs = Epoch{'s' if len(best_epochs) > 1 else ''} {', '.join(best_epochs)}")
					print(f"  After epoch {epoch} = {event}")
					last_epoch = epoch
				best_epochs = [str(e) for e in new_best_epochs if last_epoch < e]
				if best_epochs:
					print(f"  New best epochs = Epoch{'s' if len(best_epochs) > 1 else ''} {', '.join(best_epochs)}")
				print()

			print("Lowest individual encountered losses:")
			print(f"  Train loss = {min_train_loss:{loss_fmt}}")
			print(f"  Valid loss = {min_valid_loss:{loss_fmt}}")
			print(f"  Test loss  = {min_test_loss:{loss_fmt}}")
			print()

			print("Best encountered model:")
			print(f"  Train loss = {best_train_loss:{loss_fmt}}")
			print(f"  Valid loss = {best_valid_loss:{loss_fmt}}")
			print(f"  Test loss  = {best_test_loss:{loss_fmt}}")
			print()

			def calc_vs_loss_index(valid_ref=1, test_ref=1):
				return 0.5 * (len(datasets.valid) * (min_valid_loss + best_valid_loss) / valid_ref + len(datasets.test) * (min_test_loss + best_test_loss) / test_ref) / (len(datasets.valid) + len(datasets.test))

			if ref_losses:
				print("Best encountered model (relative to reference losses):")
				for ref_name, losses in ref_losses:
					print(f"  Train loss (% of {ref_name}) = {best_train_loss / losses.train:{rloss_fmt}}")
					print(f"  Valid loss (% of {ref_name}) = {best_valid_loss / losses.valid:{rloss_fmt}}")
					print(f"  Test  loss (% of {ref_name}) = {best_test_loss / losses.test:{rloss_fmt}}")
					print(f"  Valid/test loss index (% of {ref_name}) = {calc_vs_loss_index(valid_ref=losses.valid, test_ref=losses.test):{rloss_fmt}}")
				print()

			vs_loss_index = calc_vs_loss_index()
			vs_loss_index_rel = calc_vs_loss_index(valid_ref=min_ref_losses.valid, test_ref=min_ref_losses.test) if ref_losses else vs_loss_index
			print(f"Valid/test loss index: {vs_loss_index:{loss_fmt}}")
			if ref_losses:
				print(f"Valid/test loss index (% of min ref): {vs_loss_index_rel:{rloss_fmt}}")
			print()

			best_losses = dataset.DatasetTuple(best_train_loss, best_valid_loss, best_test_loss)
			R['Loss'] = {
				'min': dataset.DatasetTuple(min_train_loss, min_valid_loss, min_test_loss),
				'best': best_losses,
				'best_rel': dataset.DatasetTuple(best_train_loss / min_ref_losses.train, best_valid_loss / min_ref_losses.valid, best_test_loss / min_ref_losses.test) if ref_losses else best_losses,
				'vs_index': vs_loss_index,
				'vs_index_rel': vs_loss_index_rel,
				'vs_index_rel_pct': vs_loss_index_rel * 100,
				'vs_index_rel_str': f'{vs_loss_index_rel:{rloss_fmt}}',
			}

			if model_saves:
				print(f"Saved models:")
				output_saved_model_list = []
				for created_model_files, created_sources, created_meta_file in model_saves:
					saved_model_path = created_model_files['cs'][0]
					output_saved_model_list.append(saved_model_path)
					print(f"  {saved_model_path}")
				self.output_saved_model_list = output_saved_model_list
				self.output_saved_model = self.output_saved_model_list[-1]
				print()
			else:
				self.output_saved_model = None
				self.output_saved_model_list = [None]

			if C.TrainPerf:

				best_valid = None
				best_test = None
				best_vs = None
				RModelSaves = []

				for index, (created_model_files, created_sources, created_meta_file) in enumerate(reversed(model_saves)):

					saved_model_path, saved_model_file = created_model_files['cs']
					print(f"Evaluating performance of {saved_model_file}:")
					print("  Model metadata:")
					meta_path, meta_file, model_meta = created_meta_file
					for meta_key, meta_value in sorted(model_meta.items()):
						print(f"    {meta_key} = {meta_value}")

					self.load_model_state(model, saved_model_path, load_meta=False)

					valid_perf, valid_perf_details, valid_perf_params = model.evaluate_performance(data_loaders.valid, perf_params=None, optimise_params=C.TrainPerfOptimise)
					if best_valid is None or valid_perf > best_valid[0]:
						best_valid = (valid_perf, valid_perf_details, saved_model_file, index)
					for param, param_value in sorted(valid_perf_params.items()):
						print(f"    {param} = {param_value}")
					print("  Validation performance:")
					for perf_key, perf_value in sorted(valid_perf_details.items()):
						print(f"    {perf_key} = {perf_value:{perf_fmt(perf_value)}}")

					model_meta.update(valid_perf_params)
					self.save_model_meta(meta_path, model_meta)

					print("  Test performance:")
					test_perf, test_perf_details = model.evaluate_performance(data_loaders.test, perf_params=valid_perf_params, optimise_params=False)[:2]
					if best_test is None or test_perf > best_test[0]:
						best_test = (test_perf, test_perf_details, saved_model_file, index)
					for perf_key, perf_value in sorted(test_perf_details.items()):
						print(f"    {perf_key} = {perf_value:{perf_fmt(perf_value)}}")

					print("  Valid-test performance:")
					vs_perf, vs_perf_details = model.evaluate_performance((data_loaders.valid, data_loaders.test), perf_params=valid_perf_params, optimise_params=False)[:2]
					if best_vs is None or vs_perf > best_vs[0]:
						best_vs = (vs_perf, vs_perf_details, saved_model_file, index)
					for perf_key, perf_value in sorted(vs_perf_details.items()):
						print(f"    {perf_key} = {perf_value:{perf_fmt(perf_value)}}")

					print()

					RModelSaves.append({
						'meta': model_meta,
						'perf': {
							'valid': valid_perf,
							'valid_details': valid_perf_details,
							'test': test_perf,
							'test_details': test_perf_details,
							'vs': vs_perf,
							'vs_details': vs_perf_details,
						},
						'perf_params': valid_perf_params,
					})

					if not C.TrainPerfAllSaved:
						break

				if len(model_saves) >= 2:
					print("Best model performances:")
					print(f"  Best validation performance: {best_valid[2]} = Index {best_valid[3]}")
					for perf_key, perf_value in sorted(best_valid[1].items()):
						print(f"    {perf_key} = {perf_value:{perf_fmt(perf_value)}}")
					print(f"  Best test performance: {best_test[2]} = Index {best_test[3]}")
					for perf_key, perf_value in sorted(best_test[1].items()):
						print(f"    {perf_key} = {perf_value:{perf_fmt(perf_value)}}")
					print(f"  Best valid-test performance: {best_vs[2]} = Index {best_vs[3]}")
					for perf_key, perf_value in sorted(best_vs[1].items()):
						print(f"    {perf_key} = {perf_value:{perf_fmt(perf_value)}}")
					print()

				R['Perf'] = {  # Note: If have_info is False then the remaining fields in 'Perf' are not guaranteed to exist
					'have_info': True,
					'model_save_best': {
						'valid_index': best_valid and best_valid[3],
						'valid_perf': best_valid and best_valid[0],
						'test_index': best_test and best_test[3],
						'test_perf': best_test and best_test[0],
						'vs_index': best_vs and best_vs[3],
						'vs_perf': best_vs and best_vs[0],
					},
					'model_save_count': len(RModelSaves),
					'model_saves': RModelSaves,
				}

			else:
				R['Perf'] = {'have_info': False}

			action_cm.leave_context(high_exec_lock)

			Rsimple = self.save_results_data(R, models_subdir, subdir_name, save_pickle=C.TrainResultsPickle, save_yaml=C.TrainResultsYAML, save_json=C.TrainResultsJSON, always_simplify=True)
			self.save_results_csv(Rsimple, models_groupdir, group + '_training', CSVF)

		return R

	def eval_model_perfs_impl(self, C, saved_model_paths, force_optim=False, load_model_opts=None, load_dataset_opts=None):
		# C = Configuration of type config.Config (see also resolve_configuration() function)
		# saved_model_paths = Sequence of saved model absolute paths
		# force_optim = Whether to force optimisation of performance parameters even if already optimised ones are already available
		# Return List[Tuple[Tuple[Valid perf, Valid details, Valid loss], Tuple[Test perf, Test details, Test loss], Tuple[VS perf, VS details, VS loss]]

		C = self.resolve_configuration(C)
		self.apply_global_config(C)

		if not saved_model_paths:
			return []

		action_cm = self.ensure_entered(Context.Action)

		if load_model_opts is None:
			load_model_opts = {}
		load_model_opts['load_saved_model'] = False
		model, data_loaders = self.load_network(C, load_model_opts=load_model_opts, load_dataset_opts=load_dataset_opts)[:2]
		model.eval()

		criterion = self.load_criterion(model)

		high_exec_lock = action_cm.enter_context(self.run_lock.level(GPULevel.NormalMemHighExec))
		high_mem_exec_lock = False

		best_valid = None
		best_test = None
		best_vs = None
		outputs = []

		for saved_model_path in saved_model_paths:

			print("Loading model state:")
			print(f"  Model path: {saved_model_path}")
			saved_model_dir, saved_model_file = os.path.split(saved_model_path)
			saved_model_dir_name = os.path.basename(saved_model_dir)
			print(f"  Model file: {saved_model_file}")
			print(f"  Model dir:  {saved_model_dir_name}")
			modelC, model_meta, model_meta_path, model_meta_exists = self.load_model_state(model, saved_model_path, load_meta=True)
			if model_meta_exists:
				print(f"  Metadata YAML: {model_meta_path}")
			print(f"  Saved model configuration: {modelC.name()}")
			if model_meta:
				print("  Saved model metadata:")
				for meta_key, meta_value in sorted(model_meta.items()):
					print(f"    {meta_key} = {meta_value}")
			print()

			if high_mem_exec_lock is False:
				high_mem_exec_lock = action_cm.enter_context(self.run_lock.level(GPULevel.HighMemHighExec))

			print("Evaluating model losses:")
			valid_loss, num_valided, num_valid_batches = training.eval_model_loss(model, criterion, data_loaders.valid)
			print(f"  Validated {num_valided} samples in {num_valid_batches} batches: Mean loss {valid_loss:{loss_fmt}}")
			test_loss, num_tested, num_test_batches = training.eval_model_loss(model, criterion, data_loaders.test)
			print(f"  Tested {num_tested} samples in {num_test_batches} batches: Mean loss {test_loss:{loss_fmt}}")
			num_vsed = num_valided + num_tested
			num_vs_batches = num_valid_batches + num_test_batches
			vs_loss = (num_valided * valid_loss + num_tested * test_loss) / num_vsed
			print(f"  Valid-tested {num_vsed} samples in {num_vs_batches} batches: Mean loss {vs_loss:{loss_fmt}}")
			print()

			if not isinstance(high_mem_exec_lock, bool):
				action_cm.leave_context(high_mem_exec_lock)
				high_mem_exec_lock = True

			print("Evaluating model performance:")
			perf_params = model.resolve_performance_params(model_meta)
			perform_optimisation = force_optim or not perf_params[netmodel.META_OPTIMISED_PARAMS]
			if perform_optimisation:
				print("  Will optimise performance parameters")
			else:
				print("  Will not optimise performance parameters as model metadata already contains optimised ones")

			print("  Validation performance:")
			valid_perf, valid_perf_details, valid_perf_params = model.evaluate_performance(data_loaders.valid, perf_params=perf_params, optimise_params=perform_optimisation)
			if best_valid is None or valid_perf > best_valid[0]:
				best_valid = (valid_perf, valid_perf_details, saved_model_file)
			for perf_key, perf_value in sorted(valid_perf_details.items()):
				print(f"    {perf_key} = {perf_value:{perf_fmt(perf_value)}}")

			if perform_optimisation:
				print("  Optimised performance parameters (resaved into metadata YAML file):")
				for param_key, param_value in sorted(valid_perf_params.items()):
					print(f"    {param_key} = {param_value}")
				model_meta.update(valid_perf_params)
				self.save_model_meta(model_meta_path, model_meta)

			print("  Test performance:")
			test_perf, test_perf_details = model.evaluate_performance(data_loaders.test, perf_params=valid_perf_params, optimise_params=False)[:2]
			if best_test is None or test_perf > best_test[0]:
				best_test = (test_perf, test_perf_details, saved_model_file)
			for perf_key, perf_value in sorted(test_perf_details.items()):
				print(f"    {perf_key} = {perf_value:{perf_fmt(perf_value)}}")

			print("  Valid-test performance:")
			vs_perf, vs_perf_details = model.evaluate_performance((data_loaders.valid, data_loaders.test), perf_params=valid_perf_params, optimise_params=False)[:2]
			if best_vs is None or vs_perf > best_vs[0]:
				best_vs = (vs_perf, vs_perf_details, saved_model_file)
			for perf_key, perf_value in sorted(vs_perf_details.items()):
				print(f"    {perf_key} = {perf_value:{perf_fmt(perf_value)}}")

			print()

			outputs.append(((valid_perf, valid_perf_details, valid_loss), (test_perf, test_perf_details, test_loss), (vs_perf, vs_perf_details, vs_loss)))

		action_cm.leave_context(high_exec_lock)

		if len(saved_model_paths) >= 2:
			print("Best model performances:")
			print(f"  Best validation performance: {best_valid[2]}")
			for perf_key, perf_value in sorted(best_valid[1].items()):
				print(f"    {perf_key} = {perf_value:{perf_fmt(perf_value)}}")
			print(f"  Best test performance: {best_test[2]}")
			for perf_key, perf_value in sorted(best_test[1].items()):
				print(f"    {perf_key} = {perf_value:{perf_fmt(perf_value)}}")
			print(f"  Best valid-test performance: {best_vs[2]}")
			for perf_key, perf_value in sorted(best_vs[1].items()):
				print(f"    {perf_key} = {perf_value:{perf_fmt(perf_value)}}")
			print()

		return outputs
# EOF
