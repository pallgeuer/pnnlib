# PNN Library: Model utilities

# Imports
import hashlib

# Calculate an MD5 hash of the parameters of a model
def model_param_hash(model):
	mdsum = hashlib.md5()
	for param in model.parameters():
		mdsum.update(repr(param.dtype).encode('utf-8'))
		mdsum.update(repr(param.tolist()).encode('utf-8'))
	return mdsum.hexdigest()
# EOF
