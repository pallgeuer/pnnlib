#!/usr/bin/env python3
# Test pnnlib.transforms

# Imports
import sys
import warnings
from typing import Tuple
import PIL.Image
import PIL.ImageDraw
import numpy as np
import ppyutil.image_plot as iplt
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from pnnlib import transforms

#
# Test ResizeRect
#

# Test a particular ResizeRect transform
def RR_test(RR, image, output_size=None, num_points=11):

	RR.new_transform()
	RR.ensure_transform(input_size=image.size, output_size=output_size)

	print(f"Spatial transform: {type(RR).__name__}")
	print(RR)
	print()

	for tfrm in (None, PIL.Image.FLIP_LEFT_RIGHT, PIL.Image.FLIP_TOP_BOTTOM, PIL.Image.ROTATE_180):

		if tfrm is None:
			input_image = image
		else:
			input_image = image.transpose(tfrm)

		input_image_pre = input_image.copy()
		output_image = RR(input_image)
		if not iplt.images_equal(input_image, input_image_pre):
			raise RuntimeError("ResizeRect modified the input image in-place")
		if output_image.mode != input_image.mode:
			raise RuntimeError(f"ResizeRect: Output image has a different mode ({output_image.mode}) than the input image ({input_image.mode})")

		input_image_rgb = input_image.convert('RGB')
		output_image_rgb = iplt.ensure_rgb(output_image)

		wi, hi = input_image.size
		input_points = [(wi * (x + 0.1) / num_points, hi * (y + 0.3) / num_points) for x in range(num_points) for y in range(num_points)]
		output_points = [RR.transform_point(*p) for p in input_points]
		total_points = len(input_points)

		draw_input = PIL.ImageDraw.Draw(input_image_rgb)
		draw_output = PIL.ImageDraw.Draw(output_image_rgb)
		for i in range(total_points):
			color = f'hsl({round(900*i/total_points) % 360},100%,50%)'
			draw_input.point(input_points[i], color)
			draw_output.point(output_points[i], color)

		print(f"Input image: {input_image}")
		iplt.plot_image(input_image_rgb, show=False)
		print(f"Output image: {output_image}")
		iplt.plot_image(output_image_rgb, show=False)
		iplt.show_plots()
		print()

#
# Test CroppedResizeRect
#

# Test a particular CroppedResizeRect transform
def CRR_test(CRR, image, output_size=None, num_points=11):

	CRR.new_transform()
	CRR.ensure_transform(input_size=image.size, output_size=output_size)

	print(f"Spatial transform: {type(CRR).__name__}")
	print(CRR)
	print()

	for tfrm in (None, PIL.Image.FLIP_LEFT_RIGHT, PIL.Image.FLIP_TOP_BOTTOM, PIL.Image.ROTATE_180):

		if tfrm is None:
			input_image = image
		else:
			input_image = image.transpose(tfrm)

		input_image_pre = input_image.copy()
		output_image = CRR(input_image)
		if not iplt.images_equal(input_image, input_image_pre):
			raise RuntimeError("CroppedResizeRect modified the input image in-place")
		if output_image.mode != input_image.mode:
			raise RuntimeError(f"CroppedResizeRect: Output image has a different mode ({output_image.mode}) than the input image ({input_image.mode})")

		box: Tuple[float, float, float, float] = CRR.last_box_sized
		corners = ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3]))

		input_image_rgb = input_image.convert('RGB')
		output_image_rgb = iplt.ensure_rgb(output_image)

		wi, hi = input_image.size
		wo, ho = output_image.size

		input_points = [(wi * (x + 0.1) / num_points, hi * (y + 0.3) / num_points) for x in range(num_points) for y in range(num_points)]
		output_points = [CRR.transform_point(*p) for p in input_points]
		total_points = len(input_points)

		draw_input = PIL.ImageDraw.Draw(input_image_rgb)
		draw_output = PIL.ImageDraw.Draw(output_image_rgb)
		draw_input.polygon(corners, outline='cyan')
		# noinspection PyTypeChecker
		draw_input.line([tuple(A + 0.1*(B - A) for A, B in zip(corners[0], corners[1])), tuple(A + 0.3*(D - A) for A, D in zip(corners[0], corners[3]))], fill='lime', width=1)
		draw_output.line([(0.1*wo, 0), (0, 0.3*ho)], fill='lime', width=1)
		for i in range(total_points):
			color = f'hsl({round(900*i/total_points) % 360},100%,50%)'
			draw_input.point(input_points[i], color)
			draw_output.point(output_points[i], color)

		print(f"Input image: {input_image}")
		iplt.plot_image(input_image_rgb, show=False)
		print(f"Output image: {output_image}")
		iplt.plot_image(output_image_rgb, show=False)
		iplt.show_plots()
		print()

#
# Test RandAffineRect
#

# Test a particular RandAffineRect transform
def RAR_test(RAR, image, output_size=None, num_points=11, num_samples=4):

	RAR.new_transform()
	RAR.ensure_transform(input_size=image.size, output_size=output_size)

	print(f"Spatial transform: {type(RAR).__name__}")
	print(RAR)
	print()

	for k in range(num_samples):

		input_image = image
		input_image_pre = input_image.copy()
		output_image = RAR(input_image)
		if not iplt.images_equal(input_image, input_image_pre):
			raise RuntimeError("RandAffineRect modified the input image in-place")
		if output_image.mode != input_image.mode:
			raise RuntimeError(f"RandAffineRect: Output image has a different mode ({output_image.mode}) than the input image ({input_image.mode})")
		corners = RAR.last_corners_sized

		input_image_rgb = input_image.convert('RGB')
		output_image_rgb = iplt.ensure_rgb(output_image)

		wi, hi = input_image.size
		wo, ho = output_image.size

		input_points = [(wi * (x + 0.1) / num_points, hi * (y + 0.3) / num_points) for x in range(num_points) for y in range(num_points)]
		output_points = [RAR.transform_point(*p) for p in input_points]
		total_points = len(input_points)

		draw_input = PIL.ImageDraw.Draw(input_image_rgb)
		draw_output = PIL.ImageDraw.Draw(output_image_rgb)
		draw_input.polygon(corners, outline='cyan')
		# noinspection PyTypeChecker
		draw_input.line([tuple(A + 0.1*(B - A) for A, B in zip(corners[0], corners[1])), tuple(A + 0.3*(D - A) for A, D in zip(corners[0], corners[3]))], fill='lime', width=1)
		draw_output.line([(0.1*wo, 0), (0, 0.3*ho)], fill='lime', width=1)
		for i in range(total_points):
			color = f'hsl({round(900*i/total_points) % 360},100%,50%)'
			draw_input.point(input_points[i], color)
			draw_output.point(output_points[i], color)

		print(f"Input image: {input_image}")
		iplt.plot_image(input_image_rgb, show=False)
		print(f"Output image: {output_image}")
		iplt.plot_image(output_image_rgb, show=False)
		print(f"Full-size (with potential crop): {RAR.last_params[0]}")
		print(f"Stretch: Horizontally {'stretched' if RAR.last_params[2] >= 1 else 'squashed'} by {RAR.last_params[2]:.2f}")
		print(f"Area ratio: {RAR.last_params[3]*100:.1f}%")
		print(f"Rotation: {RAR.last_params[1]:.1f}\xB0")
		print(f"Translation: Hor {RAR.last_params[6]*100:+.1f}%, Vert {RAR.last_params[7]*100:+.1f}%")
		if RAR.last_params[4] or RAR.last_params[5]:
			print(f"Flipped: {'Hor' if RAR.last_params[4] else ''}{'Vert' if RAR.last_params[5] else ''}")
		else:
			print("Flipped: No")
		iplt.show_plots()
		print()

#
# Test RandColorJitter
#

# Test a particular RandColorJitter transform
def RCJ_test(RCJ, image):

	print(f"Color transform: {type(RCJ).__name__}")
	print(RCJ)
	print()

	print("Plotting test images...")
	utitle = "Untransformed: N/A"
	print(f"{utitle} --> {image}")
	out_image = image.convert('RGB')
	iplt.add_title(out_image, utitle)
	iplt.plot_image(out_image)

	def test_case(name, B, C, S, H):
		transform = RCJ.transform.get_params((B, B) if B is not None else None, (C, C) if C is not None else None, (S, S) if S is not None else None, (H, H) if H is not None else None)
		image_pre = image.copy()
		output_image = transform(image)
		title = f"{name}: BCSH({f'{B:.3f}' if B is not None else None}, {f'{C:.3f}' if C is not None else None}, {f'{S:.3f}' if S is not None else None}, {f'{H:.3f}' if H is not None else None})"
		print(f"{title} --> {output_image}")
		if not iplt.images_equal(image, image_pre):
			raise RuntimeError("RandColorJitter modified the input image in-place")
		if output_image.mode != image.mode:
			raise RuntimeError(f"RandColorJitter: Output image has a different mode ({output_image.mode}) than the input image ({image.mode})")
		if output_image.size != image.size:
			raise RuntimeError(f"RandColorJitter: Output image has a different size {output_image.size} than the input image {image.size}")
		output_image = output_image.convert('RGB')
		iplt.add_title(output_image, title)
		iplt.plot_image(output_image)

	test_case('Neutral', 1, 1, 1, 0)

	tfrm = RCJ.transform
	CJB = tfrm.brightness if tfrm.brightness else (None,)
	CJC = tfrm.contrast if tfrm.contrast else (None,)
	CJS = tfrm.saturation if tfrm.saturation else (None,)
	CJH = tfrm.hue if tfrm.hue else (None,)

	for Bval in CJB:
		for Cval in CJC:
			for Sval in CJS:
				for Hval in CJH:
					# Transforms are still internally applied in a random order, so show two examples of each parameter set
					test_case('All full 1', Bval, Cval, Sval, Hval)
					test_case('All full 2', Bval, Cval, Sval, Hval)

	print("Done")
	print()

#
# Test RandBriConSat
#

# Test a particular RandBriConSat transform
def RBCS_test(RBCS, image):

	print(f"Color transform: {type(RBCS).__name__}")
	print(RBCS)
	print()

	image_pre = image.copy()

	print("Plotting random parameter generation scatter and parameter edge cases...")
	edge_cases = RBCS_param_scatter(RBCS, image=image)
	print()

	print("Plotting test images...")
	title = "Untransformed: N/A"
	print(f"{title} --> {image}")
	output_image = image.convert('RGB')
	iplt.add_title(output_image, title)
	iplt.plot_image(output_image)

	output_image = RBCS.transform_image(image, params=(1, 1, 1))
	title = "Neutral: BCS(1.000, 1.000, 1.000)"
	print(f"{title} --> {output_image}")
	output_image = output_image.convert('RGB')
	iplt.add_title(output_image, title)
	iplt.plot_image(output_image)

	if not iplt.images_equal(image, image_pre):
		raise RuntimeError("RandBriConSat modified the input image in-place")

	for edge in edge_cases:
		title = f"{edge[0]}: BCS({edge[1][0]:.3f}, {edge[1][1]:.3f}, {edge[1][2]:.3f})"
		print(f"{title} --> {edge[2]}")
		if edge[2].mode != image.mode:
			raise RuntimeError(f"RandBriConSat: Output image has a different mode ({edge[2].mode}) than the input image ({image.mode})")
		if edge[2].size != image.size:
			raise RuntimeError(f"RandBriConSat: Output image has a different size {edge[2].size} than the input image {image.size}")
		out_image = edge[2].convert('RGB')
		iplt.add_title(out_image, title)
		iplt.plot_image(out_image)

	print("Done")
	print()

# Draw scatter plot of random parameter selection for all methods of RandBriConSat class object
def RBCS_param_scatter_all():
	RBCS = transforms.RandBriConSat(bri_range=(0.7, 1.8), con_range=(0.5, 1.5), sat_range=(0.6, 1.2), fixed_bcs=(0.9, 1.1, 0.8))
	# noinspection PyTypeChecker
	for method in transforms.RandBriConSatMethod:
		RBCS.set_method(method)
		RBCS_param_scatter(RBCS)

# Draw scatter plot of random parameter selection for current method of RandBriConSat class object
# noinspection PyArgumentList
def RBCS_param_scatter(RBCS, N=4000, image=None):
	# RBCS = Class object to draw parameter scatter plot for
	# N = Number of datapoints to sample and plot
	# image = Image to use when calculating edge cases
	# Return the result of RBCS.get_edge_cases(image)

	if image is None:
		image = PIL.Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype='uint8'), mode='RGB')

	edge_cases = RBCS.get_edge_cases(image)
	edgeB = [edge[1][0] for edge in edge_cases]
	edgeC = [edge[1][1] for edge in edge_cases]
	edgeS = [edge[1][2] for edge in edge_cases]

	if N < 1:
		raise ValueError("N must be at least 1")
	B = np.empty(N)
	C = np.empty(N)
	S = np.empty(N)
	for n in range(0, N):
		B[n], C[n], S[n] = RBCS.generate_params()

	if RBCS.method != transforms.RandBriConSatMethod.Fixed:
		if not RBCS.use_bri_range[0] <= B.min() <= B.max() <= RBCS.use_bri_range[1]:
			warnings.warn("Brightness has range problems")
		if not RBCS.use_con_range[0] <= C.min() <= C.max() <= RBCS.use_con_range[1]:
			warnings.warn("Contrast has range problems")
		if not RBCS.use_sat_range[0] <= S.min() <= S.max() <= RBCS.use_sat_range[1]:
			warnings.warn("Saturation has range problems")

	lims = [min(RBCS.use_bri_range[0], B.min()), max(RBCS.use_bri_range[1], B.max()), min(RBCS.use_con_range[0], C.min()), max(RBCS.use_con_range[1], C.max()), min(RBCS.use_sat_range[0], S.min()), max(RBCS.use_sat_range[1], S.max())]

	fig = plt.figure(figsize=(9, 9))
	fig.subplots_adjust(top=1, bottom=0, left=0, right=0.98)
	ax = fig.add_subplot(projection='3d')
	ax.scatter3D(B, C, S, c='b', s=10, label='Generated parameters')
	ax.scatter3D(edgeB, edgeC, edgeS, c='r', s=20, label='Edge cases')
	ax.set(xlim=lims[0:2], ylim=lims[2:4], zlim=lims[4:6], xlabel='Brightness', ylabel='Contrast', zlabel='Saturation')
	ax.set_title(repr(RBCS), pad=30)
	ax.title.set_fontsize(10)
	ax.grid(True)
	ax.legend(loc='lower left')
	plt.show()

	return edge_cases

#
# Main
#

# Main function
def main():
	RBCS_param_scatter_all()

# Run main function
if __name__ == "__main__":
	sys.exit(main())
# EOF
