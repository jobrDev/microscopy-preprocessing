# Theory behind microscopy preprocessing

This note collects the concepts that underpin the modules in this package. It
describes the imaging and signal-processing models rather than the software
interface. Throughout, an image stack has coordinates `(frame, y, x)` and
pixels are samples of a continuous image on a rectangular grid.

Preprocessing is model-dependent: it can make data more comparable and reduce
known artefacts, but it cannot recover information that was not measured. In
particular, intensity scaling, interpolation, and Fourier-domain operations
should be recorded alongside the data because they can affect quantitative
interpretation.

## `psf_otf.py`

### Point sources, the PSF, and diffraction

An ideal point emitter is represented by a delta function. The image of that
emitter is not a point: diffraction at the finite pupil of the microscope
spreads it into a **point-spread function** (PSF). In a linear,
shift-invariant approximation, an object intensity distribution `O` forms an
image

\[
I(\mathbf r) = (O * h)(\mathbf r),
\]

where `h` is the intensity PSF and `*` is convolution. The PSF therefore
sets lateral resolution and the degree to which neighbouring structures blur
into one another.

For an unaberrated circular pupil, the focal-plane field is radially symmetric
and follows a jinc function,

\[
A(r) = \frac{2J_1(v)}{v},
\]

where `J_1` is the first-order Bessel function and `v` is a dimensionless
radial coordinate. In physical coordinates it is proportional to
`NA r / lambda`, with numerical aperture `NA`, wavelength `lambda`, and
distance from the optical axis `r`. Its familiar intensity pattern is the
Airy pattern: a bright central disc with dim concentric rings. A larger NA or
shorter wavelength narrows that pattern and improves diffraction-limited
resolution.

### Amplitude PSF versus intensity PSF

The optical field is generally complex-valued. Its complex amplitude carries
both magnitude and phase, while a detector measures irradiance (intensity):

\[
I(r) = |A(r)|^2.
\]

Thus an amplitude PSF may have negative or complex oscillations, whereas an
intensity PSF is non-negative. They answer different physical questions:

| Quantity | Meaning | Typical imaging model |
| --- | --- | --- |
| Amplitude PSF `A` | Response of the optical **field** to a point source | Coherent imaging; fields add before taking magnitude squared |
| Intensity PSF `h = |A|^2` | Response of detected photon intensity to a point emitter | Incoherent fluorescence microscopy; intensities add |

For coherent illumination, interference means that the image is described by
the field response first, `E_image = E_object * A`, followed by
`I = |E_image|^2`. For spatially incoherent fluorescence, emissions from
independent fluorophores do not maintain a fixed phase relationship at the
detector, so their **intensities** add and the intensity-PSF convolution model
applies. This distinction is why squaring the magnitude is essential before
using a circular-aperture PSF as an incoherent imaging kernel.

Two normalizations are useful but distinct. Setting the peak of an amplitude
PSF to one gives a relative field profile. Setting the sum (or continuous
integral) of an intensity PSF to one makes it flux preserving: convolution
then redistributes, rather than creates or removes, total intensity away from
boundaries.

### OTF, frequency support, and sampling

The Fourier transform of the intensity PSF is the **optical transfer
function** (OTF):

\[
H(f_x,f_y) = \mathcal F\{h(x,y)\}.
\]

The image model becomes multiplication in spatial-frequency space,

\[
\mathcal F\{I\} = \mathcal F\{O\}\,H.
\]

The magnitude `|H|` is the modulation transfer function (MTF): it describes
how strongly contrast at each spatial frequency is transmitted. Low
frequencies usually pass well; high frequencies are attenuated and, beyond
the diffraction cutoff, are not transmitted. Normalizing `H(0,0)=1` fixes the
zero-frequency (mean-intensity) gain to one.

The OTF of an incoherent circular system can also be understood as the
autocorrelation of the pupil. It is not the same object as the coherent
transfer function, which is directly related to the pupil itself.

Digital arrays require bookkeeping: the discrete Fourier transform treats the
first index as the origin, while images and PSFs are commonly stored with the
optical axis at the array centre. `fftshift` and `ifftshift` only rearrange
samples to bridge those conventions; they do not change the underlying
physics. A cutoff found in an OTF array is initially a frequency-bin index.
Converting it to a physical spatial frequency requires the pixel pitch and
image dimensions. Adequate sampling also requires a pixel size fine enough to
avoid aliasing of transmitted frequencies.

### Shaping an OTF and deriving a compact kernel

Raising `|H|` to a positive power changes the weighting of spatial
frequencies before transforming back to a real-space kernel. This can be a
useful *numerical* way to choose how local or smooth a tapering kernel should
be. It should not be interpreted as changing the microscope's physical OTF:
discarding phase, exponentiating the magnitude, and cropping the resulting
kernel define a processing filter rather than a calibrated optical model.

## `background.py`

### Additive image formation and background estimation

A simple camera-image model is

\[
D_i(y,x) = S_i(y,x) + B(y,x) + N_i(y,x),
\]

where `S` is the desired specimen signal, `B` is additive background, and
`N` contains photon, detector, and read noise. Background can include camera
offset, dark current, stray light, out-of-focus fluorescence, and fixed
illumination structure. A separately acquired background stack estimates the
part that is present without the specimen signal.

Averaging independent background frames gives the pixelwise estimator

\[
\hat B(y,x) = \frac{1}{N_{bg}}\sum_j B_j(y,x).
\]

The random component of this estimate decreases approximately as
`1/sqrt(N_bg)` when frames are independent and similarly distributed.
Subtracting it produces `D_i - \hat B`. Floating-point arithmetic matters:
subtraction can create negative values, and unsigned integer arrays would
otherwise wrap those values into very large positive numbers.

The model assumes background is additive, stable during acquisition, and
registered to the measurement pixels. Uneven excitation or detector response
that acts multiplicatively is a flat-field problem, commonly modeled as
`D = g(y,x) S + B + N`, and needs gain correction in addition to (or instead
of) simple subtraction. Subtraction can also make low-signal pixels noisy or
negative; that is expected statistically and should not automatically be
clipped if quantitative noise analysis follows.

## `normalization.py`

### Global gain variation and mean-intensity normalization

Frames in a time series or image collection may differ in overall brightness
because of bleaching, exposure variation, illumination changes, or detector
gain. When the variation is well represented by one scalar per frame, a
multiplicative model is appropriate:

\[
D_i(y,x) \approx g_i S_i(y,x).
\]

Mean-intensity normalization estimates a relative gain from each frame mean
and rescales all frame means to a common reference, often the stack-wide
mean:

\[
f_i = \frac{\mu_{global}}{\mu_i}, \qquad D'_i = f_i D_i.
\]

It preserves within-frame intensity ratios but deliberately removes
between-frame changes in total signal. This is desirable for visual
consistency or algorithms sensitive to brightness, but inappropriate when
absolute fluorescence or real changes in total fluorophore amount are the
scientific endpoint. Background subtraction is normally considered first,
because additive offsets bias frame means and therefore bias the estimated
gain. Robust alternatives (for example, a median, a masked foreground mean,
or percentile-based scaling) can be better when the foreground area changes
substantially across frames.

## `drift.py`

### Lateral drift and image registration

Stage motion, thermal expansion, vibration, and sample movement can translate
successive frames even when the specimen is unchanged. With a pure lateral
translation `\Delta_i = (\Delta y_i, \Delta x_i)`, a measured frame can be
modeled as a shifted version of a reference. Registration estimates the
translation that must be applied to bring the moving frame into the reference
coordinate system. Drift correction addresses translation, not rotation,
scale change, focus drift, deformation, or biological motion; those need a
different registration model.

### Phase correlation

The Fourier shift theorem says that translating an image changes the phase of
its Fourier transform by a linear ramp while leaving its magnitude unchanged.
For reference transform `F_ref` and moving-image transform `F_mov`, the
normalized cross-power spectrum is

\[
R(u,v) = \frac{F_{ref}(u,v)F_{mov}^*(u,v)}{|F_{ref}(u,v)F_{mov}^*(u,v)|}.
\]

Its inverse transform forms a correlation surface with a peak at the relative
translation. Normalizing removes much of the influence of multiplicative
brightness differences, which is why phase correlation is often effective in
microscopy. Upsampling around the peak refines the location to subpixel
precision.

Applying a non-integer translation requires interpolation. Linear and spline
interpolation estimate values between pixels but slightly alter noise and
high-frequency content; higher order does not automatically mean more
accurate. Every shift also needs an edge convention because some output
locations map outside the measured field of view. Reliable registration
requires sufficient stable structure, reasonable signal-to-noise ratio, and
limited changes in illumination or specimen morphology. The sign convention
should always be stated: here, the estimated vector is the correction applied
to the moving frame, not merely its observed displacement.

## `edge_damping.py`

### Why image boundaries matter in Fourier processing

The discrete Fourier transform models a finite image as one tile in a
periodic image. If the left and right, or top and bottom, edges do not match,
the periodic extension has a step discontinuity. Such steps contain broad
frequency content and can produce spectral leakage and ringing, especially in
deconvolution or filtering. Boundary treatment does not add measured data; it
changes the assumption made outside the field of view.

### PSF-aware edge tapering

Edge tapering blends the original image with a blurred version near its
boundary:

\[
I_{tapered}(\mathbf r) = \alpha(\mathbf r) I(\mathbf r) +
[1-\alpha(\mathbf r)](I*h)(\mathbf r).
\]

The weight `\alpha` is close to one in the interior and falls toward zero at
the edges. The interior therefore remains close to measured data, while the
boundary transitions gradually toward a physically plausible, PSF-smoothed
continuation. A normalized PSF keeps a spatially uniform image uniform under
blurring. The taper width and shape are related to the PSF footprint: a broad
PSF spreads the transition over more pixels.

FFT-based convolution is efficient, but without padding it is circular
convolution—the source of the very wrap-around artefacts tapering is meant to
reduce. Centring a kernel in image coordinates and moving its origin to the
FFT origin are equivalent representations needed for correct discrete
convolution. Clipping a tapered result to the original intensity range limits
overshoot but may also alter quantitative values near strong structures.

### Extrapolation by normalized convolution

Another boundary strategy is to enlarge the image and infer only the added
region from pixels near the original edge. With source image `D`, binary (or
confidence) weights `W`, and an extrapolation kernel `K`, normalized
convolution uses

\[
E(\mathbf r) = \frac{(D*K)(\mathbf r)}{(W*K)(\mathbf r)}.
\]

The denominator corrects for missing samples so that the value is a weighted
average rather than one that darkens merely because fewer source pixels are
available. Restoring the original central region keeps measured pixels exact.

An inverse-distance power kernel assigns more influence to nearby boundary
samples:

\[
K(r) = \max\left(\frac{1}{r} - \frac{1}{R}, 0\right)^p.
\]

The finite cutoff `R` limits its support, and the exponent `p` controls how
quickly distant samples lose influence. This is a smooth continuation prior,
not a reconstruction of unobserved specimen detail. It is most defensible
when intensity and texture near the field boundary are locally representative;
large objects cut by the field edge or strongly directional structures can
violate that assumption.

