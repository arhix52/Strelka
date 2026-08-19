# Photometric assets

`philips_cdm_r111_35w_24deg.ies` -- a real LM-63-1995 file for a Philips
CDM-R111 35W/830 24-degree reflector lamp, from ieslibrary.com. It replaces the
synthetic cos^4 curve scene 27 used to generate.

Why a real file rather than a generated one:

* **It is a beam, not a falloff.** 3419 cd on axis dropping to 90 cd by 30
  degrees -- a 23.6-degree full beam angle recovered from the table, which
  matches the "24" in the lamp's own name and is itself a check that the table
  was read correctly. A synthetic cos^4 curve renders as a slightly vignetted
  point light, so every way of mis-reading it also renders as a slightly
  vignetted point light, which is what made the row unable to fail.
* **It exercises the parser.** 19 vertical angles by 24 horizontal planes over
  the full turn, values wrapped eight to a line, `[KEYWORD]` lines carrying
  non-ASCII bytes (a degree sign in latin-1), units type 2 and a negative
  luminaire width. The previous file had one horizontal plane and no keywords.

## Where it came from, and why that is fine

Downloaded from ieslibrary.com, which aggregates manufacturers' photometry. Its
licence page states the position plainly:

> IES and LDT files are not licensed as they can be downloaded free of charge
> from the manufacturer's website. These files are not a work of art, they are
> only the result of a light test and are provided by the manufacturer to
> simulate a light source.

(The Creative Commons terms on that page cover the library's own preview
renders, which are not used here.)

That matches what the file is: a table of measured candela per angle in a format
IES-NA standardises. Manufacturers publish it so that it can be dropped into
lighting-design and rendering tools -- which is what DIALux, Relux, AGi32 and
the sample profiles shipped with Blender and Radiance all rely on.

`install_ies()` in build_features.py falls back to a synthetic profile when this
file is absent, so removing it breaks nothing but the row's ability to catch a
mis-read table.
