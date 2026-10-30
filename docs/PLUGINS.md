# Plugins

A plugin is a JSON file in `~/Library/Application Support/GrizzyModlr/Plugins/` (Settings ▸ Tools ▸ Batch and plugins ▸ Open plugins folder).
It names a command to run on the exported current model. Pick it from **Model ▸ Tools**; nothing runs on its own.

```json
{
  "name": "Convert with assimp",
  "command": "assimp",
  "args": ["export", "{input}", "{output}"],
  "input": "glb",
  "output": "obj"
}
```

| Field | Meaning |
|---|---|
| `name` | Menu title |
| `command` | Absolute path, or a bare name looked up on PATH (+ /opt/homebrew/bin, /usr/local/bin) |
| `args` | Arguments; `{input}` exported model, `{output}` path in the run folder, `{dir}` run folder, `{name}` project name |
| `input` | Format handed to the tool: glb, obj, stl, ply, 3mf, fbx, usdz (default glb) |
| `output` | Extension of `{output}`; if the tool writes that file it is revealed in Finder |

The tool runs in a fresh temp folder; its exit status and the tail of its output are shown when it finishes.
