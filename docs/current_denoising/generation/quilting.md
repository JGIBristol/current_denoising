Quilting
====
The process of turning lots of small tiles into a larger, still decent-looking image.

This occurs in two steps - first, we take all our tiles and find the best layout to arrange
them in such that they overlap nicely along their edges.

Then, we find the best path to use to stitch them together through the overlap region.

Tile Layout
----
We need to find the best layout for putting tiles next to each other

#### Cost function
We noticed that using a simple mean square error for the overlap cost biases the algorithm towards choosing
low-energy (darker + lower-variance) tiles, which meant
that the quilt was always made of lower-noise tiles.

To fix this, I changed the cost function to use a zero-mean normalised cross-correlation.

Tile Stitching
----