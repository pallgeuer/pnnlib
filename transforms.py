# PNN Library: Data transforms

# Imports
import math
import random
import numbers
import PIL.Image
import PIL.ImageStat
from enum import auto
import torchvision.transforms
from ppyutil.classes import EnumLU

#
# Enumerations
#

# RandAffineRect rotation method enumeration
class RandAffineRectRotMethod(EnumLU):
	Uniform = auto()
	Gauss = auto()

# RandBriConSat method enumeration
class RandBriConSatMethod(EnumLU):
	Fixed = auto()
	Uniform = auto()
	Gauss = auto()

#
# Transform classes
#

# Resize rectangle transform (resizes full input size to full output size irregardless of how much stretching that involves)
class ResizeRect:

	def __init__(self, input_size=None, output_size=None, interpolation=PIL.Image.BICUBIC):
		# input_size = Initial default input size to assume
		# output_size = Initial default output size to assume
		# interpolation = Interpolation method to use
		self.input_size = input_size
		self.output_size = output_size
		self.interpolation = interpolation
		self.last_coeff_sized = None

	def __repr__(self):
		return f"{self.__class__.__name__}({_interpolation_string(self.interpolation)})"

	def __call__(self, image, output_size=None):
		# image = PIL input image to transform
		# output_size = Output size to use
		self.new_transform()
		return self.transform_image(image, output_size=output_size)

	def new_transform(self):
		pass

	def ensure_transform(self, input_size=None, output_size=None):
		# input_size = Input size to ensure
		# output_size = Output size to ensure
		transform_changed = False

		if input_size is not None and input_size != self.input_size:
			self.input_size = input_size
			transform_changed = True
		if output_size is not None and output_size != self.output_size:
			self.output_size = output_size
			transform_changed = True

		if transform_changed:
			self.last_coeff_sized = (self.output_size[0] / self.input_size[0], self.output_size[1] / self.input_size[1])

	def transform_image(self, image, output_size=None, interpolation=None):
		# image = PIL input image to transform
		# output_size = Output size to use
		# interpolation = Interpolation method to use (None => Default supplied to constructor)
		self.ensure_transform(input_size=image.size, output_size=output_size)
		return image.resize(self.output_size, resample=self.interpolation if interpolation is None else interpolation)

	def transform_pixel(self, pixelx, pixely, input_size=None, output_size=None):
		# Transforms a pixel (origin is top-left PIXEL of image) from input image to transformed image pixel coordinates
		tfrm_pointx, tfrm_pointy = self.transform_point(pixelx + 0.5, pixely + 0.5, input_size=input_size, output_size=output_size)
		return tfrm_pointx - 0.5, tfrm_pointy - 0.5

	def transform_point(self, pointx, pointy, input_size=None, output_size=None):
		# Transforms a point (origin is top-left CORNER of image) from input image to transformed image axis coordinates
		self.ensure_transform(input_size=input_size, output_size=output_size)
		return pointx * self.last_coeff_sized[0], pointy * self.last_coeff_sized[1]

# Cropped resize rectangle transform (crops and resizes the largest possible central rectangle from the input image to generate the output image without stretching)
class CroppedResizeRect:

	def __init__(self, input_size=None, output_size=None, interpolation=PIL.Image.BICUBIC):
		# input_size = Initial default input size to assume
		# output_size = Initial default output size to assume
		# interpolation = Interpolation method to use
		self.input_size = input_size
		self.output_size = output_size
		self.interpolation = interpolation
		self.last_box_unit = None
		self.last_box_sized = None
		self.new_transform_pending = True

	def __repr__(self):
		return f"{self.__class__.__name__}({_interpolation_string(self.interpolation)})"

	def __call__(self, image, output_size=None):
		# image = PIL input image to transform
		# output_size = Output size to use
		self.new_transform()
		return self.transform_image(image, output_size=output_size)

	def new_transform(self):
		self.new_transform_pending = True

	def ensure_transform(self, input_size=None, output_size=None):
		# input_size = Input size to ensure
		# output_size = Output size to ensure
		transform_changed = False

		if input_size is not None and input_size != self.input_size:
			self.input_size = input_size
			transform_changed = True
		if output_size is not None and output_size != self.output_size:
			self.output_size = output_size
			transform_changed = True

		if self.new_transform_pending:
			self._generate_unit_transform(self.input_size[0] / self.input_size[1], self.output_size[0] / self.output_size[1])  # Note: It is assumed that the input and output aspect ratios do not change (stretching occurs if it does change)
			self.new_transform_pending = False
			transform_changed = True

		if transform_changed:
			self._update_sized_transform()

	def transform_image(self, image, output_size=None, interpolation=None):
		# image = PIL input image to transform
		# output_size = Output size to use
		# interpolation = Interpolation method to use (None => Default supplied to constructor)
		self.ensure_transform(input_size=image.size, output_size=output_size)
		return image.resize(self.output_size, resample=self.interpolation if interpolation is None else interpolation, box=self.last_box_sized)

	def transform_pixel(self, pixelx, pixely, input_size=None, output_size=None):
		# Transforms a pixel (origin is top-left PIXEL of image) from input image to transformed image pixel coordinates
		tfrm_pointx, tfrm_pointy = self.transform_point(pixelx + 0.5, pixely + 0.5, input_size=input_size, output_size=output_size)
		return tfrm_pointx - 0.5, tfrm_pointy - 0.5

	def transform_point(self, pointx, pointy, input_size=None, output_size=None):
		# Transforms a point (origin is top-left CORNER of image) from input image to transformed image axis coordinates
		self.ensure_transform(input_size=input_size, output_size=output_size)
		tfrm_pointx = self.output_size[0] * (pointx - self.last_box_sized[0]) / (self.last_box_sized[2] - self.last_box_sized[0])
		tfrm_pointy = self.output_size[1] * (pointy - self.last_box_sized[1]) / (self.last_box_sized[3] - self.last_box_sized[1])
		return tfrm_pointx, tfrm_pointy

	def _generate_unit_transform(self, input_aspect, output_aspect):
		if input_aspect >= output_aspect:
			delta = (input_aspect - output_aspect) / (2 * input_aspect)
			self.last_box_unit = (delta, 0, 1 - delta, 1)
		else:
			delta = (output_aspect - input_aspect) / (2 * output_aspect)
			self.last_box_unit = (0, delta, 1, 1 - delta)

	def _update_sized_transform(self):
		wi, hi = self.input_size
		self.last_box_sized = (wi * self.last_box_unit[0], hi * self.last_box_unit[1], wi * self.last_box_unit[2], hi * self.last_box_unit[3])

# Random affine rectangle transform
class RandAffineRect:

	def __init__(self, input_size=None, output_size=None, area=(0.25, 1.0), degrees=360, rot_method=RandAffineRectRotMethod.Uniform, hor_flip=True, vert_flip=True, translate=True, stretch=1, fullsize_prob=0, interpolation=PIL.Image.BICUBIC):
		# input_size = Initial default input size to assume
		# output_size = Initial default output size to assume
		# area = Nominal range of random areas to use (where possible, NO PADDING is used) as proportions of the original image area (number = min area implies max area is 1.0, tuple = (min area, max area))
		# degrees = Range of random rotations to use (number => (-degrees, degrees), tuple => (min, max))
		# rot_method = Method to use for random rotation degree selection (uniform = uniform selection in range, gauss = gaussian selection where range specifies +-1 stddev)
		# hor_flip = Whether to include random horizontal flips in the transform
		# vert_flip = Whether to include random vertical flips in the transform
		# translate = Whether to randomly translate the rectangle or keep it centred
		# stretch = Whether to randomly stretch the image (number => maximum wider/taller stretch ratio (>=1), tuple => range of allowed stretches where >=1 means stretch apart horizontally)
		# fullsize_prob = Probability of applying a full-size transform instead of a rectangle cutout
		# interpolation = Interpolation method to use

		self.input_size = input_size
		self.output_size = output_size
		self.hor_flip = hor_flip
		self.vert_flip = vert_flip
		self.translate = translate
		self.fullsize_prob = fullsize_prob
		self.interpolation = interpolation

		if isinstance(area, tuple) and len(area) == 2:
			self.area = area
		elif isinstance(area, numbers.Real):
			self.area = (float(area), 1.0)
		else:
			raise ValueError(f"Bad area specification: {area}")
		if self.area[0] > self.area[1] or self.area[0] <= 0 or self.area[1] > 1.0:
			raise ValueError(f"Inconsistent area min/max specification: {area}")

		if isinstance(degrees, tuple) and len(degrees) == 2:
			self.degrees = degrees
		elif isinstance(degrees, (int, float)) and degrees >= 0:
			self.degrees = (-degrees, degrees)
		else:
			raise ValueError(f"Bad degrees specification: {degrees}")
		if self.degrees[0] > self.degrees[1]:
			raise ValueError(f"Inconsistent degrees min/max specification: {degrees}")

		if not isinstance(rot_method, RandAffineRectRotMethod):
			raise TypeError(f"Rotation method should be of type RandAffineRectRotMethod enum: {rot_method}")
		self.rot_method = rot_method

		if isinstance(stretch, tuple) and len(stretch) == 2:
			self.stretch = stretch
		elif isinstance(stretch, (int, float)) and stretch >= 1:
			self.stretch = (1/stretch, stretch)
		else:
			raise ValueError(f"Bad stretch specification: {stretch}")
		if self.stretch[0] > self.stretch[1] or self.stretch[0] <= 0:
			raise ValueError(f"Inconsistent stretch min/max specification: {stretch}")

		self.last_params = None
		self.last_coeff_unit = None
		self.last_invcoeff_unit = None
		self.last_corners_unit = None
		self.last_coeff_sized = None
		self.last_invcoeff_sized = None
		self.last_corners_sized = None
		self.new_transform_pending = True

	def __repr__(self):
		s = f"{self.__class__.__name__}(area={self.area[0]}->{self.area[1]}"
		if -self.degrees[0] == self.degrees[1]:
			if self.degrees[0] != 0:
				s += f", degrees={self.degrees[1]}/{self.rot_method.name}"
		else:
			s += f", degrees={self.degrees[0]}->{self.degrees[1]}/{self.rot_method.name}"
		if self.hor_flip or self.vert_flip:
			s += f", flip={'Hor' if self.hor_flip else ''}{'Vert' if self.vert_flip else ''}"
		if not self.translate:
			s += ", translate=False"
		if not self.stretch[0] == self.stretch[1] == 1:
			s += f", stretch={self.stretch[0]:.2f}->{self.stretch[1]:.2f}"
		if self.fullsize_prob > 0:
			s += f", fullsize={self.fullsize_prob}"
		s += f", {_interpolation_string(self.interpolation)})"
		return s

	def __call__(self, image, output_size=None):
		# image = PIL input image to transform
		# output_size = Output size to use
		self.new_transform()
		return self.transform_image(image, output_size=output_size)

	def new_transform(self):
		self.new_transform_pending = True

	def ensure_transform(self, input_size=None, output_size=None):
		# input_size = Input size to ensure
		# output_size = Output size to ensure
		transform_changed = False

		if input_size is not None and input_size != self.input_size:
			self.input_size = input_size
			transform_changed = True
		if output_size is not None and output_size != self.output_size:
			self.output_size = output_size
			transform_changed = True

		if self.new_transform_pending:
			self._generate_unit_transform(self.input_size[0] / self.input_size[1], self.output_size[0] / self.output_size[1])  # Note: It is assumed that the input and output aspect ratios do not change (extra stretching occurs if it does change)
			self.new_transform_pending = False
			transform_changed = True

		if transform_changed:
			self._update_sized_transform()

	def transform_image(self, image, output_size=None, interpolation=None):
		# image = PIL input image to transform
		# output_size = Output size to use
		# interpolation = Interpolation method to use (None => Default supplied to constructor)
		self.ensure_transform(input_size=image.size, output_size=output_size)
		return image.transform(self.output_size, PIL.Image.AFFINE, data=self.last_coeff_sized, resample=self.interpolation if interpolation is None else interpolation)

	def transform_pixel(self, pixelx, pixely, input_size=None, output_size=None):
		# Transforms a pixel (origin is top-left PIXEL of image) from input image to transformed image pixel coordinates
		tfrm_pointx, tfrm_pointy = self.transform_point(pixelx + 0.5, pixely + 0.5, input_size=input_size, output_size=output_size)
		return tfrm_pointx - 0.5, tfrm_pointy - 0.5

	def transform_point(self, pointx, pointy, input_size=None, output_size=None):
		# Transforms a point (origin is top-left CORNER of image) from input image to transformed image axis coordinates
		self.ensure_transform(input_size=input_size, output_size=output_size)
		tfrm_pointx = self.last_invcoeff_sized[0] * pointx + self.last_invcoeff_sized[1] * pointy + self.last_invcoeff_sized[2]
		tfrm_pointy = self.last_invcoeff_sized[3] * pointx + self.last_invcoeff_sized[4] * pointy + self.last_invcoeff_sized[5]
		return tfrm_pointx, tfrm_pointy

	def _generate_unit_transform(self, input_aspect, output_aspect):
		Ai = input_aspect

		stretch = random.uniform(self.stretch[0], self.stretch[1])  # Amount to stretch the output image (>=1 means output is widened horizontally, i.e. cutout rectangle is squashed horizontally)
		aspect = output_aspect / stretch  # Required aspect ratio of rectangle cutout = wr/hr

		fullsize = random.random() < self.fullsize_prob
		if fullsize:

			degrees = 0

			if input_aspect >= aspect:
				wd = (input_aspect - aspect) / 2
				hd = 0
			else:
				wd = 0
				hd = (aspect - input_aspect) / (2 * aspect)

			wr = input_aspect - 2 * wd
			hr = 1 - 2 * hd
			Ar = wr * hr

			if self.translate:
				translatex = random.uniform(-wd, wd)  # Amount to translate cutout rectangle horizontally from the centre of the input image
				translatey = random.uniform(-hd, hd)  # Amount to translate cutout rectangle vertically from the centre of the input image
			else:
				translatex = 0
				translatey = 0

			ABx = wr
			ABy = 0
			ADx = 0
			ADy = hr

			c = wd + translatex
			f = hd + translatey
			a = wr / output_aspect
			d = 0
			b = 0
			e = hr

		else:

			if self.rot_method == RandAffineRectRotMethod.Uniform:
				degrees = random.uniform(self.degrees[0], self.degrees[1])  # Angle to rotate cutout rectangle by (CCW)
			elif self.rot_method == RandAffineRectRotMethod.Gauss:
				degrees = random.gauss((self.degrees[0] + self.degrees[1]) / 2, (self.degrees[1] - self.degrees[0]) / 2)
			else:
				raise ValueError(f"Unrecognised rotation method: {self.rot_method}")

			radians = math.radians(degrees)
			cth = math.cos(radians)
			sth = math.sin(radians)
			Kcth = aspect * cth
			Ksth = aspect * sth
			W = max(abs(Kcth + sth), abs(Kcth - sth))  # Horizontal width of rotated cutout rectangle in units of hr
			H = max(abs(Ksth + cth), abs(Ksth - cth))  # Vertical height of rotated cutout rectangle in units of hr

			hrmax = min(input_aspect / W, 1 / H)  # Maximum value of hr so that the cutout rectangle fits cleanly without padding in the input image

			Ar = Ai * random.uniform(self.area[0], self.area[1])  # Area of cutout rectangle
			Ar = min(Ar, aspect * hrmax**2)

			hr = math.sqrt(Ar / aspect)  # Height of cutout rectangle
			wr = Ar / hr                 # Width of cutout rectangle

			if self.translate:
				translatex_max = max(input_aspect - W * hr, 0) / 2
				translatey_max = max(1 - H * hr, 0) / 2
				translatex = random.uniform(-translatex_max, translatex_max)  # Amount to translate cutout rectangle horizontally from the centre of the input image
				translatey = random.uniform(-translatey_max, translatey_max)  # Amount to translate cutout rectangle vertically from the centre of the input image
			else:
				translatex = 0
				translatey = 0

			centrex = input_aspect / 2 + translatex  # Horizontal centre of the cutout rectangle
			centrey = 0.5 + translatey               # Vertical centre of the cutout rectangle

			ABx = wr * cth  # AB is top-left to top-right corner
			ABy = -wr * sth
			ADx = hr * sth  # AD is top-left to bottom-left corner
			ADy = hr * cth

			c = centrex - (ABx + ADx) / 2
			f = centrey - (ABy + ADy) / 2
			a = ABx / output_aspect
			d = ABy / output_aspect
			b = ADx
			e = ADy

		flipH = self.hor_flip and random.random() < 0.5
		flipV = self.vert_flip and random.random() < 0.5

		if flipH:
			c += ABx
			f += ABy
			a = -a
			d = -d

		if flipV:
			c += ADx
			f += ADy
			b = -b
			e = -e

		a *= output_aspect / input_aspect
		b /= input_aspect
		c /= input_aspect
		d *= output_aspect

		self.last_params = (fullsize, degrees, stretch, Ar / Ai, flipH, flipV, translatex / input_aspect, translatey)  # Randomly selected transform parameters
		self.last_coeff_unit = (a, b, c, d, e, f)  # Affine transform coefficients (ax + by + c, dx + ey + f), where the transform is applied in axis coordinates (as opposed to pixel coordinates)

		det = a * e - b * d
		self.last_invcoeff_unit = (e / det, -b / det, (b * f - c * e) / det, -d / det, a / det, (c * d - a * f) / det)

		ac = a + c
		df = d + f
		self.last_corners_unit = ((c, f), (ac, df), (ac + b, df + e), (b + c, e + f))

	def _update_sized_transform(self):
		wi, hi = self.input_size
		wo, ho = self.output_size

		wiwo = wi / wo
		wiho = wi / ho
		hiwo = hi / wo
		hiho = hi / ho

		self.last_coeff_sized = (self.last_coeff_unit[0] * wiwo, self.last_coeff_unit[1] * wiho, self.last_coeff_unit[2] * wi, self.last_coeff_unit[3] * hiwo, self.last_coeff_unit[4] * hiho, self.last_coeff_unit[5] * hi)
		self.last_invcoeff_sized = (self.last_invcoeff_unit[0] / wiwo, self.last_invcoeff_unit[1] / hiwo, self.last_invcoeff_unit[2] * wo, self.last_invcoeff_unit[3] / wiho, self.last_invcoeff_unit[4] / hiho, self.last_invcoeff_unit[5] * ho)
		self.last_corners_sized = tuple((px * wi, py * hi) for px, py in self.last_corners_unit)

# Random color jitter transform
class RandColorJitter:

	def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
		self.transform = torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
		self.last_params = None
		self.new_transform_pending = True

	def __repr__(self):
		return "Rand" + repr(self.transform)

	def __call__(self, image):
		# image = PIL input image to transform
		# Return a new image corresponding to the input image with random color jitter applied
		self.new_transform()
		return self.transform_image(image)

	def new_transform(self):
		self.new_transform_pending = True

	def ensure_transform(self):
		if self.new_transform_pending:
			self.last_params = self.generate_params()
			self.new_transform_pending = False

	def transform_image(self, image):
		# image = PIL input image to transform
		# Return a new image corresponding to the input image with random color jitter applied
		self.ensure_transform()
		return self.last_params(image)

	def generate_params(self):
		return self.transform.get_params(brightness=self.transform.brightness, contrast=self.transform.contrast, saturation=self.transform.saturation, hue=self.transform.hue)

# Random brightness contrast saturation transform
# noinspection PyAttributeOutsideInit
class RandBriConSat:

	def __init__(self, method=RandBriConSatMethod.Uniform, bri_range=(1, 1), con_range=(1, 1), sat_range=(1, 1), fixed_bcs=None, gauss_stddevs=2, gauss_bcs_limit=0.5, gauss_bcs_limit_min=0.45):
		# method = Method of random parameter generation to use
		# bri_range = Allowed brightness range (min, max) where 0 = All black, 1 = Original, 2 = Brighter
		# con_range = Allowed contrast range (min, max) where 0 = All grey (mean grey of image), 1 = Original, 2 = Higher contrast
		# sat_range = Allowed saturation range (min, max) where 0 = Greyscale, 1 = Original, 2 = Higher saturation
		# fixed_bcs = Fixed parameters (mub, muc, mus) to use if method is Fixed, i.e. (bri_value, con_value, sat_value)
		# gauss_stddevs = Number S of standard deviations that the range limits should correspond to, i.e. range = [midpoint - S*sigma, midpoint + S*sigma]
		# gauss_bcs_limit = K in (0, 1], defines largest allowed p-norm of the normalised parameters relative to their range midpoints as the p-norm of (K, K, K)
		#                   Interpolating the three midpoints to their corresponding range limits by the factor K is the largest norm-deviation allowed away from the midpoints.
		# gauss_bcs_limit_min = Value Kmin of K that would make (1,0,0) the exact p-norm boundary between legal and not legal (it should be legal, so should have Kmin <= K)
		#                       If one parameter is at its max/min, the other two can be interpolated by a factor R (see code) towards their max/min, i.e. (1,R,R) and be on the p-norm boundary.

		self.set_method(method)
		self.set_ranges(bri_range=bri_range, con_range=con_range, sat_range=sat_range)
		self.set_fixed_bcs(fixed_bcs)

		if gauss_stddevs < 1:
			raise ValueError(f"Argument gauss_stddevs should be larger than 1: {gauss_stddevs}")
		self.gauss_stddevs = gauss_stddevs

		if not 0 < gauss_bcs_limit <= 1:
			raise ValueError(f"Argument gauss_bcs_limit should be in the range (0, 1]: {gauss_bcs_limit}")
		self.gauss_bcs_limit = gauss_bcs_limit

		if not 0 < gauss_bcs_limit_min <= gauss_bcs_limit:
			raise ValueError(f"Argument gauss_bcs_limit_min should be positive and less than gauss_bcs_limit ({gauss_bcs_limit}): {gauss_bcs_limit_min}")
		self.gauss_bcs_limit_min = gauss_bcs_limit_min

		self.gauss_p = -math.log(3) / math.log(self.gauss_bcs_limit_min)  # Solution for p to 3*Kmin^p = 1
		self.gauss_pnormp_limit = max(3 * abs(self.gauss_bcs_limit)**self.gauss_p, 1)  # If Kmin <= K <= 1, this is in [1, 3] anyway, so the max() has no effect (up to machine precision)
		self.gauss_R = ((self.gauss_pnormp_limit - 1) / 2) ** (1 / self.gauss_p)  # We rely on self.gauss_pnormp_limit >= 1

		self.last_params = None
		self.new_transform()

	def __repr__(self):
		ranges = [f"method={self.method.name}, Bri=({self.use_bri_range[0]:.4g}, {self.use_bri_range[1]:.4g}), Con=({self.use_con_range[0]:.4g}, {self.use_con_range[1]:.4g}), Sat=({self.use_sat_range[0]:.4g}, {self.use_sat_range[1]:.4g})"]
		if self.method == RandBriConSatMethod.Fixed:
			ranges.append(f"fixed_bcs={self.fixed_bcs}")
		elif self.method == RandBriConSatMethod.Gauss:
			ranges.append(f"S={self.gauss_stddevs}, K={self.gauss_bcs_limit}, Kmin={self.gauss_bcs_limit_min}, R={self.gauss_R:.2g}, p={self.gauss_p:.4g}")
		return f"{self.__class__.__name__}({', '.join(ranges)})"

	def __call__(self, image, bri_range=None, con_range=None, sat_range=None, fixed_bcs=None):
		# image = PIL input image to transform
		# bri_range, con_range, sat_range, fixed_bcs = Possible override of the ranges and fixed values used to generate the random transformation parameters
		# Return a new image corresponding to the input image with a random brightness contrast saturation transform applied
		self.new_transform(bri_range=bri_range, con_range=con_range, sat_range=sat_range, fixed_bcs=fixed_bcs)
		return self.transform_image(image)

	def set_method(self, method):
		# method = Method of random parameter generation to use
		if not isinstance(method, RandBriConSatMethod):
			raise ValueError(f"Method should be an instance of RandBriConSatMethod: {method}")
		self.method = method

	def set_ranges(self, bri_range=None, con_range=None, sat_range=None):
		# bri_range = Allowed brightness range (min, max) where 0 = All black, 1 = Original, 2 = Brighter
		# con_range = Allowed contrast range (min, max) where 0 = All grey (mean grey of image), 1 = Original, 2 = Higher contrast
		# sat_range = Allowed saturation range (min, max) where 0 = Greyscale, 1 = Original, 2 = Higher saturation

		if bri_range is not None:
			if not isinstance(bri_range, tuple) or len(bri_range) != 2:
				raise ValueError(f"Brightness range specification should be a 2-tuple: {bri_range}")
			if not 0 <= bri_range[0] <= bri_range[1]:
				raise ValueError(f"Brightness range should be positive and ascending: {bri_range}")
			self.bri_range = bri_range

		if con_range is not None:
			if not isinstance(con_range, tuple) or len(con_range) != 2:
				raise ValueError(f"Contrast range specification should be a 2-tuple: {con_range}")
			if not 0 <= con_range[0] <= con_range[1]:
				raise ValueError(f"Contrast range should be positive and ascending: {con_range}")
			self.con_range = con_range

		if sat_range is not None:
			if not isinstance(sat_range, tuple) or len(sat_range) != 2:
				raise ValueError(f"Saturation range specification should be a 2-tuple: {sat_range}")
			if not 0 <= sat_range[0] <= sat_range[1]:
				raise ValueError(f"Saturation range should be positive and ascending: {sat_range}")
			self.sat_range = sat_range

	def set_fixed_bcs(self, fixed_bcs):
		# fixed_bcs = Fixed parameters (mub, muc, mus) to use if method is Fixed, i.e. (bri_value, con_value, sat_value)
		if not isinstance(fixed_bcs, tuple) and len(fixed_bcs) == 3:
			raise ValueError(f"Argument fixed_bcs should be a 3-tuple of parameter values: {fixed_bcs}")
		if any(mu < 0 for mu in fixed_bcs):
			raise ValueError(f"Argument fixed_bcs should only contain positive values: {fixed_bcs}")
		self.fixed_bcs = fixed_bcs

	def new_transform(self, bri_range=None, con_range=None, sat_range=None, fixed_bcs=None):
		# bri_range, con_range, sat_range, fixed_bcs = Possible override of the ranges and fixed values used to generate the random transformation parameters
		self.use_bri_range = bri_range or self.bri_range
		self.use_con_range = con_range or self.con_range
		self.use_sat_range = sat_range or self.sat_range
		self.use_bri_midpoint = (self.use_bri_range[0] + self.use_bri_range[1]) / 2
		self.use_con_midpoint = (self.use_con_range[0] + self.use_con_range[1]) / 2
		self.use_sat_midpoint = (self.use_sat_range[0] + self.use_sat_range[1]) / 2
		self.use_bri_amplitude = (self.use_bri_range[1] - self.use_bri_range[0]) / 2
		self.use_con_amplitude = (self.use_con_range[1] - self.use_con_range[0]) / 2
		self.use_sat_amplitude = (self.use_sat_range[1] - self.use_sat_range[0]) / 2
		self.use_fixed_bcs = fixed_bcs or self.fixed_bcs
		self.new_transform_pending = True

	def ensure_transform(self):
		if self.new_transform_pending:
			self.last_params = self.generate_params()
			self.new_transform_pending = False

	def transform_image(self, image, params=None):
		# image = PIL input image to transform
		# params = Manual set of parameters (mub, muc, mus) to use
		# Return a new image corresponding to the input image with the required brightness contrast saturation transform applied

		if params is None:
			self.ensure_transform()
			params = self.last_params

		mub, muc, mus = params
		if mub < 0 or muc < 0 or mus < 0:
			raise ValueError(f"Mu parameters must be non-negative: ({mub}, {muc}, {mus})")

		mode = image.mode
		greyscale = (mode == 'L')
		convert = (mode != 'RGB' and not greyscale)

		if convert:
			image = image.convert('RGB')

		scaleI = mub * muc
		if scaleI != 1:
			image_out = image.point(lambda p: round(p * scaleI))  # Note: Image.point() clamps at [0, 255]
		else:
			image_out = image

		if mus != 1 or muc != 1:
			image_grey = image if greyscale else image.convert('L')
			if mus != 1:
				mubc = (2 + 2*mub*muc + mub + muc) / 6
				image_grey_full = image_grey if greyscale else image_grey.convert('RGB')
				sat_part = image_grey_full.point(lambda p: round(p * mubc))  # Note: Image.point() clamps at [0, 255]
				image_out = PIL.Image.blend(sat_part, image_out, mus)
			if muc != 1:
				mubs = (2 + 2*mub*mus + mub + mus) / 6
				mean_grey = PIL.ImageStat.Stat(image_grey).mean[0]
				offset = round((1 - muc) * mubs * mean_grey)  # Note: Equivalent to rounding inside the lambda on the next line as p is uint8
				image_out = image_out.point(lambda p: p + offset)  # Note: Image.point() clamps at [0, 255]

		if convert:
			image_out = image_out.convert(mode)

		return image_out

	def generate_params(self):
		# Return a random set of parameters (mub, muc, mus) (without modifying anything in the class)

		if self.method == RandBriConSatMethod.Fixed:
			mub, muc, mus = self.use_fixed_bcs

		elif self.method == RandBriConSatMethod.Gauss:
			Sinv = 1 / self.gauss_stddevs
			mubhat = random.gauss(0, Sinv)
			muchat = random.gauss(0, Sinv)
			mushat = random.gauss(0, Sinv)

			muhat_pnormp = abs(mubhat)**self.gauss_p + abs(muchat)**self.gauss_p + abs(mushat)**self.gauss_p
			if muhat_pnormp > self.gauss_pnormp_limit:
				scale_factor = (self.gauss_pnormp_limit / muhat_pnormp) ** (1 / self.gauss_p)
				mubhat *= scale_factor
				muchat *= scale_factor
				mushat *= scale_factor

			mub = min(max(self.use_bri_midpoint + mubhat * self.use_bri_amplitude, self.use_bri_range[0]), self.use_bri_range[1])
			muc = min(max(self.use_con_midpoint + muchat * self.use_con_amplitude, self.use_con_range[0]), self.use_con_range[1])
			mus = min(max(self.use_sat_midpoint + mushat * self.use_sat_amplitude, self.use_sat_range[0]), self.use_sat_range[1])

		elif self.method == RandBriConSatMethod.Uniform:
			mub = random.uniform(self.use_bri_range[0], self.use_bri_range[1])
			muc = random.uniform(self.use_con_range[0], self.use_con_range[1])
			mus = random.uniform(self.use_sat_range[0], self.use_sat_range[1])

		else:
			raise ValueError(f"Unrecognised RandBriConSat method: {self.method}")

		return mub, muc, mus

	def get_edge_cases(self, image, method=None):
		# image = PIL input image to transform to all of the edge cases
		# method = Method of random parameter generation to generate the edge cases for (default = use method saved in class)
		# Return a list of tuples (type, param, output image)

		if method is None:
			method = self.method

		edge_cases = []
		plus_minus = (-1, 1)
		midpoint = (self.use_bri_midpoint, self.use_con_midpoint, self.use_sat_midpoint)
		amplitude = (self.use_bri_amplitude, self.use_con_amplitude, self.use_sat_amplitude)

		def add_edge_case(case_type, mubcs):
			edge_cases.append((case_type, mubcs, self.transform_image(image, params=mubcs)))

		if method != RandBriConSatMethod.Fixed:
			add_edge_case('Centre', midpoint)
			add_edge_case('Pure brightness', (self.use_bri_range[0], midpoint[1], midpoint[2]))
			add_edge_case('Pure brightness', (self.use_bri_range[1], midpoint[1], midpoint[2]))
			add_edge_case('Pure contrast', (midpoint[0], self.use_con_range[0], midpoint[2]))
			add_edge_case('Pure contrast', (midpoint[0], self.use_con_range[1], midpoint[2]))
			add_edge_case('Pure saturation', (midpoint[0], midpoint[1], self.use_sat_range[0]))
			add_edge_case('Pure saturation', (midpoint[0], midpoint[1], self.use_sat_range[1]))

		if method == RandBriConSatMethod.Fixed:
			add_edge_case('Fixed', self.use_fixed_bcs)

		elif method == RandBriConSatMethod.Uniform:
			for b in plus_minus:
				for c in plus_minus:
					for s in plus_minus:
						sign = (b, c, s)
						add_edge_case('All full', tuple(M + s*H for s, M, H in zip(sign, midpoint, amplitude)))

		elif method == RandBriConSatMethod.Gauss:
			K = self.gauss_bcs_limit
			R = self.gauss_R
			for b in plus_minus:
				for c in plus_minus:
					for s in plus_minus:
						sign = (b, c, s)
						add_edge_case('Equal parts', tuple(M + s*H*P for s, M, H, P in zip(sign, midpoint, amplitude, (K, K, K))))
						add_edge_case('Full brightness', tuple(M + s*H*P for s, M, H, P in zip(sign, midpoint, amplitude, (1, R, R))))
						add_edge_case('Full contrast', tuple(M + s*H*P for s, M, H, P in zip(sign, midpoint, amplitude, (R, 1, R))))
						add_edge_case('Full saturation', tuple(M + s*H*P for s, M, H, P in zip(sign, midpoint, amplitude, (R, R, 1))))

		else:
			raise ValueError(f"Unrecognised RandBriConSat method: {method}")

		return edge_cases

#
# Helper functions
#

def _interpolation_string(interpolation):
	# interpolation = Interpolation method from the PIL library
	if interpolation == PIL.Image.BICUBIC:
		return "BICUBIC"
	elif interpolation == PIL.Image.BILINEAR:
		return "BILINEAR"
	elif interpolation == PIL.Image.NEAREST:
		return "NEAREST"
	else:
		return f"interp={interpolation}"
# EOF
