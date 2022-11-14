# bevy_scene_sdf

dynamically create a cascade of 128^3 sdf textures from world mesh data, based on 512^3 voxelization of the scene.

references:
https://twitter.com/guitio2002/status/1577590023888723970
https://twitter.com/reduzio/status/1583780332524146692

uses a bevy fork with #5703 plus a couple of fields made public on ExtractedAssets.

simple example:
`cargo run --example simple -- X` with X = a positive integer will create a grid of cubes
`cargo run --example simple` will try to load bistro.glb from assets

after the scene is loaded you'll probably need to press `8` to force recalc the sdfs.
`9` + `0` will rescale the sdfs up or down by 10%
mouse + wasd to move

issues:
- doesn't allow you to specify what is dynamic and what is static, it just takes all the mesh data in the world
- uses way too much vram (we can't allocate on gpu with wgsl so i create large buffers for intermediate data, at least one of which is totally unnecessary)
- jfa stitch doesn't do the merging/stitching on updates quite right, so i worked around it by scaling distances down in output
- fine raster needs some work still
  - only does 1 axis optimisation for voxel fine raster (should do 3, and should choose iteration order by triangle normal's dominant axis)
  - doesn't include SAT tests in the axis reduction so still needs calcs per voxel
