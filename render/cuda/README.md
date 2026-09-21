# Hand-written CUDA backend

This backend implements the repository's sphere path tracer directly in CUDA.

- `prepare()` converts the scene to single-precision device data and builds a median-split binary BVH on the host. The preorder BVH uses escape links, so ray traversal does not allocate a per-thread stack. The scene and BVH are uploaded once and reused for every frame.
- A CUDA thread renders one pixel and accumulates its samples, tracing Lambertian, metal, and dielectric paths through exact sphere intersections.
- CUDA events measure kernel execution. Host wall time also includes camera setup, launch, synchronization, and copying the RGBA image back to the host.

The implementation is specialized for the current static sphere scene and camera orbit. Geometry updates, meshes, textures, and other material types are outside this backend's current scope.
