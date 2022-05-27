# PNN Library: YAML specification loading

# Imports
from enum import Enum, auto
from collections import namedtuple
import yaml
import util.print

# Join lists enumeration
class JoinLists(Enum):
	Disabled = auto()
	Prepend = auto()
	Append = auto()

# YAML spec named tuple
YAMLSpec = namedtuple('YAMLSpec', 'name spec')  # Format: str, any valid parsed YAML

#
# Classes
#

# YAML specification error class
class YAMLSpecError(Exception):

	def __init__(self, yaml_file, message):
		self.yaml_file = yaml_file
		self.message = message

	def __str__(self):
		return f"In file '{self.yaml_file}': {self.message}"

# YAML specification manager class
class YAMLSpecManager:

	def __init__(self, yaml_file, merge=True, merge_depth=0, keep_inherited=False, join_lists=JoinLists.Disabled, default_inheritance=True, default_name='Default', inherits_key='$inherits'):
		# yaml_file = File path of YAML file to load
		# merge = Whether to resolve inheritance using a merge strategy (as opposed to a replacement strategy)
		# merge_depth = Depth to apply the merge strategy to, if enabled, before switching to a replacement strategy (0 => Merge at all depths)
		# keep_inherited = Whether the replacement strategy keeps the older inherited value (True) or the newer derived value (False)
		# join_lists = Whether to use a join strategy for lists instead of replacement (JoinLists enum)
		# default_inheritance = Whether all specifications by default inherit from the default specification, unless explicitly otherwise specified
		# default_name = Name of the default specification
		# inherits_key = String key to interpret as specifying spec inheritance

		self.yaml_file = yaml_file
		self.merge = merge
		self.merge_depth = merge_depth
		self.keep_inherited = keep_inherited
		self.join_lists = join_lists if isinstance(join_lists, JoinLists) else JoinLists.Disabled
		self.default_inheritance = default_inheritance
		self.default_name = default_name
		self.inherits_key = inherits_key

		self.__spec_dict = self.__parse_spec()

	def __parse_spec(self):

		try:
			with open(self.yaml_file, 'r') as file:
				yaml_data = yaml.load(file, Loader=yaml.CSafeLoader)
		except OSError as e:
			raise YAMLSpecError(self.yaml_file, f"Failed to open YAML spec file: {e}")
		except yaml.YAMLError as e:
			raise YAMLSpecError(self.yaml_file, f"Failed to parse YAML spec file: {e}")

		if not isinstance(yaml_data, dict):
			return {self.default_name: yaml_data}

		yaml_data = {name: spec for name, spec in yaml_data.items() if isinstance(name, str)}

		def topo_process_spec(sname):
			local_seen.add(sname)
			sdata = yaml_data[sname]
			cname = None
			if isinstance(sdata, dict):
				cname = sdata.get(self.inherits_key, self.default_name if self.default_inheritance and sname != self.default_name else None)
				if cname in yaml_data:
					if cname in local_seen:
						raise YAMLSpecError(self.yaml_file, f"Found an inheritance cycle involving the YAML spec: {cname}")
					if cname not in spec_dict:
						topo_process_spec(cname)
				elif cname is not None:
					raise YAMLSpecError(self.yaml_file, f"YAML spec {sname} inherits from the non-existent YAML spec: {cname} ({type(cname).__name__})")
			spec_dict[sname] = YAMLSpec(sname, self.__compute_specification(sdata, spec_dict.get(cname, None)))

		spec_dict = {}
		for name, spec in yaml_data.items():
			if name not in spec_dict:
				local_seen = set()
				topo_process_spec(name)

		return spec_dict

	def __compute_specification(self, sdata, cdatat):

		if isinstance(cdatat, YAMLSpec):

			def resolve_data(base, new, depth=0):
				if isinstance(base, dict) and isinstance(new, dict) and self.merge and (self.merge_depth <= 0 or depth < self.merge_depth):
					merged = {}
					for key in set(base).union(new):
						if key in base and key in new:
							merged[key] = resolve_data(base[key], new[key], depth=depth+1)
						elif key in new:
							merged[key] = new[key]
						else:
							merged[key] = base[key]
					return merged
				elif isinstance(base, list) and isinstance(new, list) and self.join_lists != JoinLists.Disabled:
					return new + base if self.join_lists == JoinLists.Prepend else base + new
				elif self.keep_inherited:
					return base.copy() if depth == 0 and isinstance(base, dict) else base
				else:
					return new.copy() if depth == 0 and isinstance(new, dict) else new

			sdata = resolve_data(cdatat.spec, sdata)
			if isinstance(sdata, dict):
				sdata.pop(self.inherits_key, None)

		else:

			if isinstance(sdata, dict) and self.inherits_key in sdata:
				sdata = sdata.copy()
				del sdata[self.inherits_key]

		return sdata

	def spec_dict(self):
		return self.__spec_dict

	def spec_list(self):
		return list(self.__spec_dict.values())

	def spec_names(self):
		return list(self.__spec_dict)

	def get_spec(self, name):
		if name not in self.__spec_dict:
			raise YAMLSpecError(self.yaml_file, f"YAML spec not found: {name}")
		return self.__spec_dict[name]

	def get_spec_(self, name, default=None):
		return self.__spec_dict.get(name, default)

	def default_spec(self):
		return self.get_spec(self.default_name)

	def default_spec_(self, default=None):
		return self.get_spec_(self.default_name, default=default)

	def pprint(self, header='YAML spec'):
		iprint = util.print.PrefixedPrinter('  ')
		for name, spec in self.__spec_dict.values():
			print(f"{header} {name}:")
			yaml.dump(spec, stream=iprint, indent=2, sort_keys=True, allow_unicode=True, default_flow_style=False, width=2147483647, Dumper=yaml.CSafeDumper)
			print()
# EOF
