# grizzymodlr design

## product

grizzymodlr is a native macos app for turning one image into a 3d mesh.

local first. no image uploads. shape and paint run on device through mlx swift.

one project has:

- source image
- optional mask
- shape mesh versions
- optional painted mesh versions
- export files

main flow:

1. open or drop an image
2. remove or edit the background
3. generate a shape
4. paint the shape if needed
5. export glb, obj, stl, or ply

## app shape

one main window. swiftui views on the main actor.

AppState holds app state. AppReducer is the only place that changes it.

events go into the reducer. the reducer returns new state plus effects.

effects do downloads, model loading, generation, painting, cancel, and file writes.

engines send progress and results back as events.

main pieces:

- AppState, AppEvent, AppEffect, AppReducer
- ProjectStore for projects and versions
- ModelStore for installed models
- DownloadManager for resumable downloads
- EngineArbiter for one gpu job at a time
- ShapeEngine for shape generation
- PaintEngine for texture generation
- MeshExporter for glb, obj, stl, and ply
- MeshDecimator before uv unwrap

## models

two shape models:

- small: hunyuan3d-dit-v2-mini
- large: hunyuan3d-dit-v2-0-turbo

two paint models:

- small: hunyuan3d-paint-v2-0
- large: hunyuan3d-paintpbr-v2-1

ModelCatalog contains repo ids, revisions, file sizes, and sha256 values.

models download on first use. downloads resume from partial files. completed files get size and hash checks.

only one model job owns the gpu. switching from shape to paint releases the other model's weights.

## state and failure rules

job states stay small and visible in the ui:

- idle
- preparing
- waiting for engine
- loading model
- conditioning or rendering
- denoising
- decoding
- meshing or baking
- committing
- cancelling
- failed

every job has a token. stale callbacks get ignored.

cancel removes staged files and returns the project to idle.

failed jobs show the stage and message. retry starts a new run.

if a model is missing, send the user to model management. no dead-end error dialog.

## storage

app data lives under `~/Library/Application Support/Modelr/`.

```text
models/
  shape-small/
  shape-large/
  paint-small/
  paint-large/
projects.json
projects/<uuid>/
```

projects save source images, masks, meshes, textures, and version metadata.

writes use temp files plus atomic rename. corrupt project indexes get backed up and rebuilt from project folders.

## ui

toolbar groups:

- history
- shape model and quality
- paint model and quality

project view shows the image and the active 3d result.

onboarding offers fast start, best quality, everything, or skip.

model management supports install, pause, resume, retry, import, and remove.

all copy stays lowercase except grizzymodlr, class names, model ids, file paths, and protocol names.

## verification

headless tests cover the reducer, downloads, project storage, mesh export, and mesh decimation.

test flow:

1. create project
2. import image
3. generate shape
4. paint texture
5. export glb
6. exit with success
