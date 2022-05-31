# PNN Library: Configuration parameters

# Imports
import re
import sys
import ast
import inspect
import collections
import configparser
from enum import Enum
from ppyutil.classes import EnumLU

#
# Classes
#

# Configuration error class
class ConfigError(Exception):
	pass

# Config class
class Config:

	__inited = False

	def __init__(self, name, cro, local_configs, resolved_configs):
		# name = Configuration name
		# cro = Ordered tuple of configuration resolution order (CRO) names (starting from greatest ancestor)
		# local_configs = Dict of local configuration parameters
		# resolved_configs = Dict of resolved configuration parameters
		self.__name = name
		self.__cro = cro
		self.__local_dict = dict(sorted(local_configs.items()))
		self.__resolved_dict = dict(sorted(resolved_configs.items()))
		for key, value in self.__resolved_dict.items():
			setattr(self, key, value)
		self.__modified = False
		self.__inited = True

	def __setattr__(self, key, value):
		if self.__inited:
			super().__setattr__('_Config__modified', True)
			self.__resolved_dict[key] = value
		super().__setattr__(key, value)

	def name(self, strict=False):
		if not strict and self.__modified:
			return self.__name + '*'
		else:
			return self.__name

	def cro(self):
		return self.__cro

	def local_dict(self):
		return self.__local_dict

	def dict(self):
		return self.__resolved_dict

	def keys(self):
		return set(self.__resolved_dict.keys())

	def _asdict(self):
		return self.__resolved_dict.copy()

	def __repr__(self):
		return "Config({0}, {1})".format(self.name(), ', '.join('{0}: {1}'.format(key, f"'{value}'" if isinstance(value, str) else value) for key, value in self.__resolved_dict.items()))

	def pprint(self, newline=True):
		print(f"Configuration {self.name()}:")
		for key, value in self.__resolved_dict.items():
			if isinstance(value, Enum):
				valuep = value.name
			elif isinstance(value, set):
				valuep = value and f"{{{', '.join(repr(v) for v in sorted(value))}}}"
			else:
				valuep = value
			print(f"  {key} = {valuep}")
		if newline:
			print()

# Config manager class
class ConfigManager:

	def __init__(self, config_file, config_spec, config_check=None, converters=None, default_name='Default', **kwargs):
		# config_file = File path to load configurations from
		# config_spec = Dict that contains all expected config parameters and maps them to class types (e.g. to str, bool, int, float, MyEnum)
		# config_check = Specification of allowed ranges/check functions for the configs (scalar = min, tuple = (min, max), else func(x) returning bool where true means config is okay)
		# converters = Iterable of converters to use (in addition to the pre-defined bool, int, float), specified as a list of EnumLU/tuple subclasses and/or (class, converter) tuples
		#              (converter is a custom converter function that must always return an object of type 'class', multiple converters to the same class are not supported)
		# default_name = Expected name of the default parameter section in the config file
		# kwargs = Extra arguments to pass to the constructor of the internal ConfigParser object

		self.config_spec = config_spec
		self.config_spec_set = set(self.config_spec)
		self.config_check = config_check if config_check is not None else {}
		self.config_check_set = set(self.config_check)
		self.def_name = default_name

		if not self.config_check_set.issubset(self.config_spec_set):
			raise ConfigError(f"Configs to check includes parameters not found in spec: {', '.join(sorted(self.config_check_set - self.config_spec_set))}")

		supported_converters = get_module_enums(__name__)  # Defining enums inside this module is supported => They are automatically added as converters here

		literal_eval = lambda string: ast.literal_eval(string)  # noqa
		supported_converters.append(('boolean', convert_to_boolean))
		supported_converters.append((int.__name__, int))
		supported_converters.append((float.__name__, float))
		supported_converters.append((str.__name__, str))
		supported_converters.append((tuple.__name__, literal_eval))
		supported_converters.append((list.__name__, literal_eval))
		supported_converters.append((set.__name__, lambda string: ast.literal_eval(string) or set()))

		if converters is not None:
			for converter in converters:
				num_converters = len(supported_converters)
				if isinstance(converter, tuple) and len(converter) == 2:
					supported_converters.append((converter[0].__name__, converter[1]))
				elif inspect.isclass(converter):
					if issubclass(converter, EnumLU):
						supported_converters.append((converter.__name__, converter.from_str))
					elif issubclass(converter, tuple):
						# noinspection PyProtectedMember
						supported_converters.append((converter.__name__, lambda string: converter._make(ast.literal_eval(string))))
				if len(supported_converters) == num_converters:
					raise ConfigError(f"Invalid converter specification: {converter}")

		supported_converters = {name: converter for name, converter in supported_converters}

		kwargs['default_section'] = object()
		kwargs.setdefault('allow_no_value', False)
		kwargs.setdefault('strict', True)
		kwargs.setdefault('empty_lines_in_values', False)
		kwargs.setdefault('interpolation', configparser.ExtendedInterpolation())
		self.__parser = configparser.ConfigParser(converters=supported_converters, **kwargs)
		self.__parser.optionxform = str

		read_files = self.__parser.read(config_file)
		if isinstance(config_file, str):
			if len(read_files) != 1:
				raise ConfigError(f"Configuration file not found: {config_file}")
		else:
			if len(read_files) != len(config_file):
				raise ConfigError(f"Configuration file(s) not found: {', '.join(set(config_file).difference(read_files))}")

		self.__config_dict = {}
		for raw_name, section in self.__parser.items():
			if raw_name == self.__parser.default_section:
				continue
			name, parents, local_configs = self._parse_config_name(raw_name, name_spec=True)
			for key, value_str in section.items():
				local_configs[key] = self._parse_value(key, value_str)
			self.__config_dict[name] = self._generate_config(name, parents, local_configs)

		if self.def_name not in self.__config_dict:
			raise ConfigError(f"Default configuration name is not a key in the config dict: {self.def_name}")

	def default_name(self):
		return self.def_name

	def default_config(self):
		return self.__config_dict[self.def_name]

	def config_names(self):
		return list(self.__config_dict)

	def config_dict(self):
		return self.__config_dict

	def get_config(self, name, strict=False):
		config = self.__config_dict.get(name, None)
		if config is not None:
			return config
		elif strict:
			raise ConfigError(f"Configuration not found: {name}")
		return self._generate_config(*self._parse_config_name(name, name_spec=False))

	def pprint(self):
		for config in self.__config_dict.values():
			config.pprint()

	def _parse_value(self, key, value_str):
		ctype = self.config_spec.get(key, str)
		if ctype == bool:
			converter_name = 'boolean'
		elif inspect.isclass(ctype):
			converter_name = ctype.__name__
		else:
			raise ConfigError(f"Config parameter {key} has a spec type that is not a class: {ctype}")
		converter = self.__parser.converters.get(converter_name, None)
		if converter is None:
			raise ConfigError(f"No converter found for spec type: {ctype.__name__}")
		value = converter(value_str)
		if not isinstance(value, ctype):
			raise ConfigError(f"Value parsed for key {key} is not of spec type {ctype.__name__}: {type(value)}")
		return value

	def _parse_config_name(self, raw_name, name_spec):
		if name_spec:
			match = re.fullmatch(r'^\s*([^<+?=&\s]+)\s*(?:<\s*([^<+?=&\s]+\s*(?:\+\s*[^<+?=&\s]+\s*)*))?(?:\?\s*([^<+?=&\s]+\s*=\s*[^<=&\s][^<=&]*\s*(?:&\s*[^<+?=&\s]+\s*=\s*[^<=&\s][^<=&]*\s*)*))?$', raw_name)
			if not match:
				raise ConfigError(f"Failed to parse named configuration: {raw_name}")
			name, raw_parents, raw_configs = match.groups()
		else:
			match = re.fullmatch(r'^\s*([^<+?=&\s]+\s*(?:\+\s*[^<+?=&\s]+\s*)*)(?:\?\s*([^<+?=&\s]+\s*=\s*[^<=&\s][^<=&]*\s*(?:&\s*[^<+?=&\s]+\s*=\s*[^<=&\s][^<=&]*\s*)*))?$', raw_name)
			if not match:
				raise ConfigError(f"Failed to parse unnamed configuration: {raw_name}")
			raw_parents, raw_configs = match.groups()
			name = None

		if raw_parents is None:
			parents = (self.def_name,) if name != self.def_name else ()
		else:
			if name == self.def_name:
				raise ConfigError("The default configuration cannot have any parents")
			parents = tuple(parent.strip() for parent in raw_parents.split('+'))
			parent_counts = collections.Counter(parents)
			if any(count >= 2 for count in parent_counts.values()):
				raise ConfigError(f"Cannot specify same parent configuration more than once: {raw_name}")
			elif parent_counts[self.def_name] != 0 and parents[0] != self.def_name:
				raise ConfigError(f"If the default configuration is explicitly specified as a parent then it must be the first one: {raw_name}")

		local_configs = {}
		local_config_strs = {}
		if raw_configs is not None:
			for assignment in raw_configs.split('&'):
				key, value_str = assignment.split('=')
				key = key.strip()
				value_str = value_str.strip()
				local_config_strs[key] = value_str
				local_configs[key] = self._parse_value(key, value_str)

		if name is None:
			name = '+'.join(parents)
			if local_config_strs:
				name = f"{name}?{'&'.join(f'{key}={value_str}' for key, value_str in local_config_strs.items())}"

		return name, parents, local_configs

	def _generate_config(self, name, parents, local_configs):
		if name in self.__config_dict:
			raise ConfigError(f"Configuration of name {name} already exists")

		cro_seqs = [[name]]
		for parent_name in reversed(parents):
			parent_config = self.__config_dict.get(parent_name, None)
			if parent_config is None:
				raise ConfigError(f"Failed to find configuration parent: {parent_name}")
			cro_seqs.append(list(parent_config.cro()))
		cro_seqs.append(list(parents))

		cro = []
		while True:
			cro_seqs = [seq for seq in cro_seqs if seq]
			if not cro_seqs:
				break
			for seq in cro_seqs:
				if any(seq[-1] in s[:-1] for s in cro_seqs):
					continue
				else:
					next_ancestor = seq[-1]
					break
			else:
				raise ConfigError(f"Unresolvable configuration resolution order for configuration: {name}")
			cro.append(next_ancestor)
			for seq in cro_seqs:
				if seq[-1] == next_ancestor:
					seq.pop()
		cro.reverse()

		resolved_configs = {}
		for ancestor_name in cro[:-1]:
			ancestor_config = self.__config_dict.get(ancestor_name, None)
			if ancestor_config is None:
				raise ConfigError(f"Failed to find configuration ancestor: {ancestor_name}")
			resolved_configs.update(ancestor_config.local_dict())
		resolved_configs.update(local_configs)

		config = Config(name, cro, local_configs, resolved_configs)

		valid, msgs = self._check_config(config)
		if not valid:
			raise ConfigError(f"Configuration manager detected config parameter problems:\n" + '\n'.join(msgs))

		return config

	@staticmethod
	def _check_config_spec(msgs, config_str, value, spec):
		if isinstance(spec, (int, float)) and not isinstance(spec, bool):  # bool is a subclass of int!
			if value < spec:
				msgs.append(f"Value does not meet specification for {config_str}: {value} (need >={spec})")
		elif isinstance(spec, tuple) and len(spec) == 2:
			if not spec[0] <= value <= spec[1]:
				msgs.append(f"Value does not meet specification for {config_str}: {value} (need [{spec[0]}, {spec[1]}])")
		elif callable(spec):
			check_result = spec(value)
			if isinstance(check_result, bool):
				if not check_result:
					msgs.append(f"Value does not meet specification for {config_str}: {value} (determined by function)")
			else:
				msgs.append(f"Invalid value check function for {config_str}, output was: {check_result} (need bool)")
		else:
			msgs.append(f"Invalid value check specification for {config_str}: {spec}")
		return None

	def _check_config(self, config):
		msgs = []
		cname = config.name()
		try:
			config_keys = config.keys()
			config_dict = config.dict()

			if config_keys != self.config_spec_set:
				missing_vars = self.config_spec_set.difference(config_keys)
				extra_vars = config_keys.difference(self.config_spec_set)
				msgs.append(f"Configuration {cname} is not one-to-one with the config spec!")
				if missing_vars:
					msgs.append(f"Configuration {cname} is missing variables: {', '.join(sorted(missing_vars))}")
				if extra_vars:
					msgs.append(f"Configuration {cname} has extra variables: {', '.join(sorted(extra_vars))}")

			for pname, pvalue in config_dict.items():
				if pname not in self.config_spec:
					continue
				ctype = self.config_spec[pname]
				if not isinstance(pvalue, ctype):
					msgs.append(f"Configuration {cname} param {pname} does not have the expected type ({type(pvalue).__name__} instead of {ctype.__name__})")

			for pname, pcheck in self.config_check.items():
				if pname not in self.config_spec or pname not in config_dict:
					continue
				ctype = self.config_spec[pname]
				pvalue = config_dict[pname]
				config_name = f"{cname}->{pname}"
				if issubclass(ctype, (tuple, list, set)):
					if isinstance(pcheck, list):
						self._check_config_spec(msgs, f"len({config_name})", len(pvalue), pcheck[0])
						for v in pvalue:
							self._check_config_spec(msgs, config_name, v, pcheck[1])
					else:
						self._check_config_spec(msgs, f"len({config_name})", len(pvalue), pcheck)
				else:
					self._check_config_spec(msgs, config_name, pvalue, pcheck)

		except (TypeError, ValueError, LookupError):
			raise ConfigError("Configuration manager detected fatal config parameter problems:\n" + '\n'.join(msgs))

		return not msgs, msgs

#
# Helper functions
#

# Convert a string to boolean
def convert_to_boolean(value):
	# noinspection PyProtectedMember
	return configparser.ConfigParser._convert_to_boolean(configparser.ConfigParser, value)  # Note: Dirty trick to call a method of ConfigParser that has a static implementation, but is declared as an instance method

# Get a list of all EnumLU classes in a module
def get_module_enums(module_name):
	# module_name = String name of module to search, e.g. __name__ to search the calling one
	enum_tuple_list = inspect.getmembers(sys.modules[module_name], lambda member: inspect.isclass(member) and member.__module__ == module_name and issubclass(member, EnumLU))
	return [class_ for name, class_ in enum_tuple_list]
# EOF
