# PNN Library: CUDA driver API utilities

# Imports
import threading
import pycuda.driver

# Module variables
_initialised = False
_initialisation_lock = threading.Lock()

# Initialise the CUDA driver API if it hasn't been initialised so far by this module
def ensure_initialised():
	global _initialised
	if _initialised:
		return
	with _initialisation_lock:
		if _initialised:
			return
		# noinspection PyUnresolvedReferences
		pycuda.driver.init()
		_initialised = True

# Return whether the CUDA driver API has been initialised
def is_initialised():
	return _initialised
# EOF
