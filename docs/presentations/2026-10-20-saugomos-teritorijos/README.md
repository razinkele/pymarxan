# pymarxan — saugomų teritorijų administracijoms (2026-10-20)

17-slide Lithuanian deck for protected-area administrations: what pymarxan does, real-life uses,
development status at v0.45.0, and the features not yet built. Live-talk format; the talking points
are in the speaker notes.

| File | What it is |
|---|---|
| `pymarxan_saugomos_teritorijos.pptx` | The deck |
| `build.js` | pptxgenjs generator for the deck |
| `map.py` | Slide 6 map (`baltic_map.png`) from `examples/baltic_marine_planning.py` |
| `mosaic_data.py` | Slide 8 reed-bed mosaic data (`mosaic.json`) from `examples/curonian_spit_reedbeds.py` |
| `rivers_lt.py` | Slide 9 figures (`*_lt.png`) — Lithuanian labels over `scripts/make_rivers_figures.py` |
| `rivers_tab_crop.png` | Slide 10 screenshot, cropped from `docs/images/rivers_tab.png` |

## Rebuild

```bash
# figures (shiny micromamba env — needs pymarxan + PuLP)
/opt/micromamba/envs/shiny/bin/python map.py
/opt/micromamba/envs/shiny/bin/python rivers_lt.py
/opt/micromamba/envs/shiny/bin/python mosaic_data.py

# deck: needs a node_modules with pptxgenjs, react, react-dom, react-icons, sharp,
# and apply_theme.js from the pptx skill (writes the theme colours into the file)
NODE_PATH=/path/to/node_modules APPLY_THEME=/path/to/pptx/scripts/apply_theme.js node build.js
```

The "undeveloped features" slides (13–14) and the status numbers (slide 4) describe v0.45.0;
update them as features ship.
