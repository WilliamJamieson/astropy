# astropy.modeling Model Class Inventory

See also: [Concrete Public Model Inventory](MODEL_CLASS_INVENTORY_CONCRETE.md)

This document lists model classes defined in `astropy.modeling` source files (excluding tests).

Method: static AST scan + inheritance closure from `Model`/`FittableModel`/projection/polynomial/tabular/spline model roots.

- Total detected model classes: **148**
- Public model classes: **140**
- Internal/base model classes: **8**

## Downstream Explicit Model Usage (GitHub Survey Snapshot)

This snapshot summarizes explicit usage of specific `astropy.modeling` model
classes observed in downstream packages.

Requested packages:

- `spacetelescope/gwcs` (confidence: high)
	Source: `AffineTransformation2D`, `Const1D`, `Identity`, `Mapping`,
	`Pix2Sky_TAN`, `Polynomial2D`, `RotateCelestial2Native`, `Scale`, `Shift`
	Test-only: `Identity`, `Mapping`, `Pix2Sky_Gnomonic`, `Polynomial2D`,
	`Scale`, `Shift`
- `spacetelescope/jwst` (confidence: high)
	Source: `AffineTransformation2D`, `Const1D`, `Identity`, `Linear1D`,
	`Mapping`, `Planar2D`, `Polynomial1D`, `Polynomial2D`,
	`RotationSequence3D`, `Scale`, `Shift`, `Tabular1D`
	Test-only: `Const1D`, `Identity`, `Mapping`, `Multiply`, `Planar2D`,
	`Polynomial1D`, `Polynomial2D`, `RotationSequence3D`, `Scale`, `Shift`
- `spacetelescope/romancal` (confidence: high)
	Source: `Identity`, `Mapping`, `RotationSequence3D`, `Scale`, `Shift`,
	`Spline1D`
	Test-only: `Gaussian2D`, `models` namespace usage
- `spacetelescope/romanisim` (confidence: medium)
	Source: `Mapping`, `Pix2Sky_TAN`, `Polynomial2D`
	Test-only: no explicit specific-model imports found in this survey run
- `spacetelescope/stpsf` (confidence: medium)
	Source: `Gaussian1D` (from `astropy.modeling.functional_models`)
	Test-only: no explicit specific-model imports found in this survey run
- `astropy/photutils` (confidence: high)
	Source: `Const2D`, `Gaussian1D`, `Gaussian2D`, `Identity`, `Moffat1D`,
	`Moffat2D`, `Polynomial2D`, `Shift`
	Test-only: `Const2D`, `Gaussian1D`, `Gaussian2D`, `Moffat1D`, `Moffat2D`,
	`Polynomial2D`
- `sunpy/sunpy` (confidence: low)
	Source: no direct source-level explicit model imports found in this survey run
	Test-only: no explicit specific-model imports found in this survey run
	Note: SunPy ecosystem package `sunpy/ndcube` explicitly uses `Tabular1D`
	and `Tabular2D`.
- `DKISTDC/dkist` (confidence: medium)
	Source: no explicit specific-model imports observed in non-test paths in this
	survey run
	Test-only: `AffineTransformation2D`, `Multiply`, `Pix2Sky_TAN`,
	`RotateNative2Celestial`, `Shift`, `Tabular1D`
- `spacetelescope/stdatamodels` (confidence: high)
	Source: `Const1D`, `Mapping`, `Rotation2D`, `Tabular1D`
	Test-only: `Const1D`, `Identity`, `Mapping`, `Polynomial1D`,
	`Polynomial2D`, `Rotation2D`, `Shift`
- `spacetelescope/roman_datamodels` (confidence: low)
	Source: no direct source-level explicit specific-model imports found in this
	survey run
	Test-only: no explicit specific-model imports found in this survey run
- `spacetelescope/stcal` (confidence: medium)
	Source: no explicit specific-model imports observed in non-test paths in this
	survey run
	Test-only: `Scale`, `Shift`
- `spacetelescope/stpipe` (confidence: low)
	Source: no direct source-level explicit specific-model imports found in this
	survey run
	Test-only: no explicit specific-model imports found in this survey run

Other `spacetelescope` organization packages with explicit specific-model usage:

- `spacetelescope/synphot_refactor` (confidence: medium)
	Source: `RedshiftScaleFactor`, `Scale`, `Sine1D`
	Test-only: not surveyed in detail
- `spacetelescope/mirage` (confidence: medium)
	Source: `BlackBody`, `Gaussian1D`, `Lorentz1D`, `Mapping`, `Polynomial2D`,
	`Sersic1D`, `Sersic2D`, `Shift`, `Voigt1D`
	Test-only: not surveyed in detail
- `spacetelescope/tweakwcs` (confidence: medium)
	Source: `AffineTransformation2D`, `Const1D`, `Identity`
	Test-only: not surveyed in detail
- `spacetelescope/astrocut` (confidence: low)
	Source: `SIP`
	Test-only: not surveyed in detail
- `spacetelescope/jwreftools` (confidence: medium)
	Source: `Identity`, `Mapping`, `Polynomial2D`, `Shift`
	Test-only: not surveyed in detail

Notes:

- This is a code-search snapshot and is not guaranteed to be exhaustive.
- Some repositories primarily use `astropy.modeling` via module aliases
	(for example `from astropy.modeling import models`) or indirect APIs.

## Source-Only Popularity Ranking (Most to Least Used)

Ranking basis: number of distinct downstream packages in this survey with
explicit non-test usage of each model class.

### Tier 1 (very widely used)

- `Mapping` — 7 packages
- `Identity` — 6 packages
- `Polynomial2D` — 6 packages
- `Shift` — 6 packages

### Tier 2 (widely used)

- `Const1D` — 4 packages
- `Scale` — 4 packages

### Tier 3 (common)

- `AffineTransformation2D` — 3 packages
- `Gaussian1D` — 3 packages

### Tier 4 (recurring, specialized)

- `Pix2Sky_TAN` — 2 packages
- `RotationSequence3D` — 2 packages
- `Tabular1D` — 2 packages

### Tier 5 (package-specific)

- `BlackBody`, `Const2D`, `Gaussian2D`, `Linear1D`, `Lorentz1D`, `Moffat1D`,
	`Moffat2D`, `Planar2D`, `Polynomial1D`, `RedshiftScaleFactor`,
	`RotateCelestial2Native`, `Rotation2D`, `SIP`, `Sersic1D`, `Sersic2D`,
	`Sine1D`, `Spline1D`, `Voigt1D` — 1 package each

### Machine-Readable Usage Table (CSV)

```csv
model,package_count,package_list
Mapping,7,"gwcs;jwreftools;jwst;mirage;romancal;romanisim;stdatamodels"
Identity,6,"gwcs;jwreftools;jwst;photutils;romancal;tweakwcs"
Polynomial2D,6,"gwcs;jwreftools;jwst;mirage;photutils;romanisim"
Shift,6,"gwcs;jwreftools;jwst;mirage;photutils;romancal"
Const1D,4,"gwcs;jwst;stdatamodels;tweakwcs"
Scale,4,"gwcs;jwst;romancal;synphot_refactor"
AffineTransformation2D,3,"gwcs;jwst;tweakwcs"
Gaussian1D,3,"mirage;photutils;stpsf"
Pix2Sky_TAN,2,"gwcs;romanisim"
RotationSequence3D,2,"jwst;romancal"
Tabular1D,2,"jwst;stdatamodels"
BlackBody,1,"mirage"
Const2D,1,"photutils"
Gaussian2D,1,"photutils"
Linear1D,1,"jwst"
Lorentz1D,1,"mirage"
Moffat1D,1,"photutils"
Moffat2D,1,"photutils"
Planar2D,1,"jwst"
Polynomial1D,1,"jwst"
RedshiftScaleFactor,1,"synphot_refactor"
RotateCelestial2Native,1,"gwcs"
Rotation2D,1,"stdatamodels"
SIP,1,"astrocut"
Sersic1D,1,"mirage"
Sersic2D,1,"mirage"
Sine1D,1,"synphot_refactor"
Spline1D,1,"romancal"
Voigt1D,1,"mirage"
```

### Documentation Corroboration (Package Docs)

- `gwcs` docs (for example `docs/gwcs/constructing_gwcs_models.rst`,
	`docs/gwcs/points_to_wcs.rst`) explicitly describe compound modeling,
	projections, and polynomial distortion fitting.
- `jwst` docs (`docs/jwst/assign_wcs/asdf-howto.rst`) explicitly frame WCS
	reference construction around `astropy.modeling` models relevant to assign_wcs.
- `photutils` docs (for example `docs/user_guide/isophote.rst`,
	`docs/user_guide/radial_profiles.rst`, `docs/whats_new/3.0.rst`) explicitly
	use and discuss `Gaussian2D`, `Moffat1D`, and fitter-backed model workflows.
- `romancal` docs (`docs/roman/references_general/distortion_reffile.rst`)
	explicitly specify distortion payloads as `astropy.modeling.Model`.

## Public Model Classes

### `astropy.modeling.convolution`

- `Convolution` (bases: `CompoundModel`)

### `astropy.modeling.core`

- `CompoundModel` (bases: `Model`)
- `Fittable1DModel` (bases: `FittableModel`)
- `Fittable2DModel` (bases: `FittableModel`)
- `FittableModel` (bases: `Model`)

### `astropy.modeling.functional_models`

- `AiryDisk2D` (bases: `Fittable2DModel`)
- `ArcCosine1D` (bases: `_InverseTrigonometric1D`)
- `ArcSine1D` (bases: `_InverseTrigonometric1D`)
- `ArcTangent1D` (bases: `_InverseTrigonometric1D`)
- `Box1D` (bases: `Fittable1DModel`)
- `Box2D` (bases: `Fittable2DModel`)
- `Const1D` (bases: `Fittable1DModel`)
- `Const2D` (bases: `Fittable2DModel`)
- `Cosine1D` (bases: `_Trigonometric1D`)
- `Disk2D` (bases: `Fittable2DModel`)
- `Ellipse2D` (bases: `Fittable2DModel`)
- `Exponential1D` (bases: `Fittable1DModel`)
- `Gaussian1D` (bases: `Fittable1DModel`)
- `Gaussian2D` (bases: `Fittable2DModel`)
- `GeneralSersic2D` (bases: `Sersic2D`)
- `KingProjectedAnalytic1D` (bases: `Fittable1DModel`)
- `Linear1D` (bases: `Fittable1DModel`)
- `Logarithmic1D` (bases: `Fittable1DModel`)
- `Lorentz1D` (bases: `Fittable1DModel`)
- `Lorentz2D` (bases: `Fittable2DModel`)
- `Moffat1D` (bases: `Fittable1DModel`)
- `Moffat2D` (bases: `Fittable2DModel`)
- `Multiply` (bases: `Fittable1DModel`)
- `Planar2D` (bases: `Fittable2DModel`)
- `RedshiftScaleFactor` (bases: `Fittable1DModel`)
- `RickerWavelet1D` (bases: `Fittable1DModel`)
- `RickerWavelet2D` (bases: `Fittable2DModel`)
- `Ring2D` (bases: `Fittable2DModel`)
- `Scale` (bases: `Fittable1DModel`)
- `Sersic1D` (bases: `Fittable1DModel`)
- `Sersic2D` (bases: `Fittable2DModel`)
- `Shift` (bases: `Fittable1DModel`)
- `Sine1D` (bases: `_Trigonometric1D`)
- `Tangent1D` (bases: `_Trigonometric1D`)
- `Trapezoid1D` (bases: `Fittable1DModel`)
- `TrapezoidDisk2D` (bases: `Fittable2DModel`)
- `Voigt1D` (bases: `Fittable1DModel`)

### `astropy.modeling.mappings`

- `Identity` (bases: `Mapping`)
- `Mapping` (bases: `FittableModel`)
- `UnitsMapping` (bases: `Model`)

### `astropy.modeling.physical_models`

- `BlackBody` (bases: `Fittable1DModel`)
- `Drude1D` (bases: `Fittable1DModel`)
- `NFW` (bases: `Fittable1DModel`)
- `Plummer1D` (bases: `Fittable1DModel`)

### `astropy.modeling.polynomial`

- `Chebyshev1D` (bases: `_PolyDomainWindow1D`)
- `Chebyshev2D` (bases: `OrthoPolynomialBase`)
- `Hermite1D` (bases: `_PolyDomainWindow1D`)
- `Hermite2D` (bases: `OrthoPolynomialBase`)
- `InverseSIP` (bases: `Model`)
- `Legendre1D` (bases: `_PolyDomainWindow1D`)
- `Legendre2D` (bases: `OrthoPolynomialBase`)
- `OrthoPolynomialBase` (bases: `PolynomialBase`)
- `Polynomial1D` (bases: `_PolyDomainWindow1D`)
- `Polynomial2D` (bases: `PolynomialModel`)
- `PolynomialBase` (bases: `FittableModel`)
- `PolynomialModel` (bases: `PolynomialBase`)
- `SIP` (bases: `Model`)

### `astropy.modeling.powerlaws`

- `BrokenPowerLaw1D` (bases: `Fittable1DModel`)
- `ExponentialCutoffPowerLaw1D` (bases: `Fittable1DModel`)
- `LogParabola1D` (bases: `Fittable1DModel`)
- `PowerLaw1D` (bases: `Fittable1DModel`)
- `Schechter1D` (bases: `Fittable1DModel`)
- `SmoothlyBrokenPowerLaw1D` (bases: `Fittable1DModel`)

### `astropy.modeling.projections`

- `AffineTransformation2D` (bases: `Model`)
- `Conic` (bases: `Projection`)
- `Cylindrical` (bases: `Projection`)
- `HEALPix` (bases: `Projection`)
- `Pix2SkyProjection` (bases: `Projection`)
- `Pix2Sky_Airy` (bases: `Pix2SkyProjection, Zenithal`)
- `Pix2Sky_BonneEqualArea` (bases: `Pix2SkyProjection, PseudoConic`)
- `Pix2Sky_COBEQuadSphericalCube` (bases: `Pix2SkyProjection, QuadCube`)
- `Pix2Sky_ConicEqualArea` (bases: `Pix2SkyProjection, Conic`)
- `Pix2Sky_ConicEquidistant` (bases: `Pix2SkyProjection, Conic`)
- `Pix2Sky_ConicOrthomorphic` (bases: `Pix2SkyProjection, Conic`)
- `Pix2Sky_ConicPerspective` (bases: `Pix2SkyProjection, Conic`)
- `Pix2Sky_CylindricalEqualArea` (bases: `Pix2SkyProjection, Cylindrical`)
- `Pix2Sky_CylindricalPerspective` (bases: `Pix2SkyProjection, Cylindrical`)
- `Pix2Sky_Gnomonic` (bases: `Pix2SkyProjection, Zenithal`)
- `Pix2Sky_HEALPix` (bases: `Pix2SkyProjection, HEALPix`)
- `Pix2Sky_HEALPixPolar` (bases: `Pix2SkyProjection, HEALPix`)
- `Pix2Sky_HammerAitoff` (bases: `Pix2SkyProjection, PseudoCylindrical`)
- `Pix2Sky_Mercator` (bases: `Pix2SkyProjection, Cylindrical`)
- `Pix2Sky_Molleweide` (bases: `Pix2SkyProjection, PseudoCylindrical`)
- `Pix2Sky_Parabolic` (bases: `Pix2SkyProjection, PseudoCylindrical`)
- `Pix2Sky_PlateCarree` (bases: `Pix2SkyProjection, Cylindrical`)
- `Pix2Sky_Polyconic` (bases: `Pix2SkyProjection, PseudoConic`)
- `Pix2Sky_QuadSphericalCube` (bases: `Pix2SkyProjection, QuadCube`)
- `Pix2Sky_SansonFlamsteed` (bases: `Pix2SkyProjection, PseudoCylindrical`)
- `Pix2Sky_SlantOrthographic` (bases: `Pix2SkyProjection, Zenithal`)
- `Pix2Sky_SlantZenithalPerspective` (bases: `Pix2SkyProjection, Zenithal`)
- `Pix2Sky_Stereographic` (bases: `Pix2SkyProjection, Zenithal`)
- `Pix2Sky_TangentialSphericalCube` (bases: `Pix2SkyProjection, QuadCube`)
- `Pix2Sky_ZenithalEqualArea` (bases: `Pix2SkyProjection, Zenithal`)
- `Pix2Sky_ZenithalEquidistant` (bases: `Pix2SkyProjection, Zenithal`)
- `Pix2Sky_ZenithalPerspective` (bases: `Pix2SkyProjection, Zenithal`)
- `Projection` (bases: `Model`)
- `PseudoConic` (bases: `Projection`)
- `PseudoCylindrical` (bases: `Projection`)
- `QuadCube` (bases: `Projection`)
- `Sky2PixProjection` (bases: `Projection`)
- `Sky2Pix_Airy` (bases: `Sky2PixProjection, Zenithal`)
- `Sky2Pix_BonneEqualArea` (bases: `Sky2PixProjection, PseudoConic`)
- `Sky2Pix_COBEQuadSphericalCube` (bases: `Sky2PixProjection, QuadCube`)
- `Sky2Pix_ConicEqualArea` (bases: `Sky2PixProjection, Conic`)
- `Sky2Pix_ConicEquidistant` (bases: `Sky2PixProjection, Conic`)
- `Sky2Pix_ConicOrthomorphic` (bases: `Sky2PixProjection, Conic`)
- `Sky2Pix_ConicPerspective` (bases: `Sky2PixProjection, Conic`)
- `Sky2Pix_CylindricalEqualArea` (bases: `Sky2PixProjection, Cylindrical`)
- `Sky2Pix_CylindricalPerspective` (bases: `Sky2PixProjection, Cylindrical`)
- `Sky2Pix_Gnomonic` (bases: `Sky2PixProjection, Zenithal`)
- `Sky2Pix_HEALPix` (bases: `Sky2PixProjection, HEALPix`)
- `Sky2Pix_HEALPixPolar` (bases: `Sky2PixProjection, HEALPix`)
- `Sky2Pix_HammerAitoff` (bases: `Sky2PixProjection, PseudoCylindrical`)
- `Sky2Pix_Mercator` (bases: `Sky2PixProjection, Cylindrical`)
- `Sky2Pix_Molleweide` (bases: `Sky2PixProjection, PseudoCylindrical`)
- `Sky2Pix_Parabolic` (bases: `Sky2PixProjection, PseudoCylindrical`)
- `Sky2Pix_PlateCarree` (bases: `Sky2PixProjection, Cylindrical`)
- `Sky2Pix_Polyconic` (bases: `Sky2PixProjection, PseudoConic`)
- `Sky2Pix_QuadSphericalCube` (bases: `Sky2PixProjection, QuadCube`)
- `Sky2Pix_SansonFlamsteed` (bases: `Sky2PixProjection, PseudoCylindrical`)
- `Sky2Pix_SlantOrthographic` (bases: `Sky2PixProjection, Zenithal`)
- `Sky2Pix_SlantZenithalPerspective` (bases: `Sky2PixProjection, Zenithal`)
- `Sky2Pix_Stereographic` (bases: `Sky2PixProjection, Zenithal`)
- `Sky2Pix_TangentialSphericalCube` (bases: `Sky2PixProjection, QuadCube`)
- `Sky2Pix_ZenithalEqualArea` (bases: `Sky2PixProjection, Zenithal`)
- `Sky2Pix_ZenithalEquidistant` (bases: `Sky2PixProjection, Zenithal`)
- `Sky2Pix_ZenithalPerspective` (bases: `Sky2PixProjection, Zenithal`)
- `Zenithal` (bases: `Projection`)

### `astropy.modeling.rotations`

- `EulerAngleRotation` (bases: `_EulerRotation, Model`)
- `RotateCelestial2Native` (bases: `_SkyRotation`)
- `RotateNative2Celestial` (bases: `_SkyRotation`)
- `Rotation2D` (bases: `Model`)
- `RotationSequence3D` (bases: `Model`)
- `SphericalRotationSequence` (bases: `RotationSequence3D`)

### `astropy.modeling.spline`

- `Spline1D` (bases: `_Spline`)

## Internal/Base Model Classes

### `astropy.modeling.functional_models`

- `_InverseTrigonometric1D` (bases: `_Trigonometric1D`)
- `_Trigonometric1D` (bases: `Fittable1DModel`)

### `astropy.modeling.math_functions`

- `_NPUfuncModel` (bases: `Model`)

### `astropy.modeling.polynomial`

- `_PolyDomainWindow1D` (bases: `PolynomialModel`)
- `_SIP1D` (bases: `PolynomialBase`)

### `astropy.modeling.rotations`

- `_SkyRotation` (bases: `_EulerRotation, Model`)

### `astropy.modeling.spline`

- `_Spline` (bases: `FittableModel`)

### `astropy.modeling.tabular`

- `_Tabular` (bases: `Model`)
