"""Extensive help content for every application page.

Each key maps to a :class:`shiny.ui.TagList` that is shown inside a modal
dialog when the user clicks the **ⓘ Help** button in the card header.
"""
from __future__ import annotations

from shiny import ui

# Helpers for consistent formatting ----------------------------------------

def _section(title: str) -> ui.Tag:
    return ui.h5(title, class_="mt-3 mb-2 text-primary")


def _ul(*items: str) -> ui.Tag:
    return ui.tags.ul(*[ui.tags.li(i) for i in items], class_="mb-2")


def _tip(text: str) -> ui.Tag:
    return ui.div(
        ui.tags.strong("💡 Tip: "),
        text,
        class_="alert alert-info py-2 px-3 small",
    )


def _param_table(rows: list[tuple[str, str]]) -> ui.Tag:
    header = ui.tags.thead(
        ui.tags.tr(
            ui.tags.th("Parameter", scope="col"),
            ui.tags.th("Description", scope="col"),
        )
    )
    body = ui.tags.tbody(
        *[
            ui.tags.tr(ui.tags.td(ui.tags.strong(name)), ui.tags.td(desc))
            for name, desc in rows
        ]
    )
    return ui.tags.table(
        header, body,
        class_="table table-sm table-bordered mb-3",
    )


# --------------------------------------------------------------------------
# Content dictionary
# --------------------------------------------------------------------------

HELP_CONTENT: dict[str, ui.TagList] = {

    # =====================================================================
    # DATA INPUT
    # =====================================================================
    "upload": ui.TagList(
        _section("Overview"),
        ui.p(
            "This page lets you load an existing Marxan project so the "
            "application can display, configure, and solve it. A Marxan "
            "project is a directory containing a configuration file and "
            "several data tables."
        ),
        _section("Required Project Files"),
        _ul(
            "input.dat — the master configuration file. It lists the paths "
            "to all other input files and sets default solver parameters "
            "(BLM, NUMITNS, etc.).",
            "pu.dat — planning unit table with columns: id, cost, and "
            "status. Each row is one planning unit.",
            "spec.dat (or spf.dat) — conservation feature table with "
            "columns: id, target, spf, name. Each row is one feature "
            "(species, habitat, ecosystem service).",
            "puvspr.dat — planning-unit-vs-species matrix. Columns: "
            "species (feature id), pu (planning unit id), amount.",
            "bound.dat (optional) — boundary lengths between adjacent "
            "planning units. Used for the spatial compactness penalty.",
        ),
        _section("How to Use"),
        ui.tags.ol(
            ui.tags.li(
                ui.tags.strong("Upload a ZIP: "),
                "Click 'Upload Marxan project (.zip)' and select a ZIP file "
                "containing a complete project directory. The application "
                "extracts it, locates input.dat, and loads all referenced files."
            ),
            ui.tags.li(
                ui.tags.strong("Browse a server directory: "),
                "Use the directory browser to navigate the file system. "
                "Folders that contain an input.dat file are marked with a "
                "green 'Marxan project' badge. Select the folder and click "
                "'Load selected directory'."
            ),
        ),
        _section("What Happens After Loading"),
        ui.p(
            "Once loaded, the project summary shows the number of planning "
            "units, features, and the solver parameters read from input.dat. "
            "You can now proceed to any other tab — Features to edit targets, "
            "Configure to choose a solver, or Run to produce solutions."
        ),
        _tip(
            "If loading fails, check that input.dat uses relative paths and "
            "that all referenced files exist inside the project directory."
        ),
    ),

    # -----------------------------------------------------------------
    "import_wizard": ui.TagList(
        _section("Overview"),
        ui.p(
            "Import planning units directly from a geospatial file instead "
            "of loading a pre-built Marxan project. This is useful when you "
            "have your own shapefile or GeoJSON of planning regions."
        ),
        _section("Supported Formats"),
        _ul(
            "Shapefile (.shp) — must include .shx, .dbf, and .prj sidecar files",
            "GeoJSON (.geojson)",
            "GeoPackage (.gpkg)",
        ),
        _section("Column Mapping"),
        _param_table([
            ("ID Column", "Unique identifier for each planning unit (becomes Marxan's PUID). Must be an integer column."),
            ("Cost Column", "Numeric cost of including each PU in the reserve (e.g. area, land price, opportunity cost)."),
            ("Status Column", "Optional Marxan status code: 0 = available, 1 = initial include, 2 = locked in, 3 = locked out."),
        ]),
        _section("How to Use"),
        ui.tags.ol(
            ui.tags.li("Upload a geospatial file using the file input."),
            ui.tags.li("The application reads the file and populates the column dropdowns with the attribute table columns."),
            ui.tags.li("Select the correct ID, cost, and (optionally) status columns."),
            ui.tags.li("Click 'Import' to create a new Marxan problem from the geometry."),
        ),
        ui.p(
            "After import, adjacency boundaries are automatically computed "
            "from polygon geometry to create the equivalent of bound.dat."
        ),
        _tip(
            "Ensure your file has a defined CRS (coordinate reference system). "
            "For area-based costs, project to an equal-area CRS before importing."
        ),
    ),

    # -----------------------------------------------------------------
    "grid_builder": ui.TagList(
        _section("Overview"),
        ui.p(
            "Create a regular grid of planning units over a geographic "
            "extent. This is the fastest way to set up a Marxan problem "
            "when you don't have pre-existing planning unit polygons."
        ),
        _section("Parameters"),
        _param_table([
            ("Min X / Max X", "Western and eastern bounds (longitude in decimal degrees for geographic CRS)."),
            ("Min Y / Max Y", "Southern and northern bounds (latitude in decimal degrees)."),
            ("Cell Size", "Width and height of each grid cell in coordinate units. Smaller cells = finer resolution but more planning units."),
            ("Grid Type", "Square or Hexagonal. Hexagonal grids minimise edge effects and provide more uniform neighbour distances — preferred for ecological planning."),
            ("Clip to GADM", "If checked, cells outside the GADM administrative boundary (fetched on the Spatial tab) are removed, restricting the grid to your study area."),
        ]),
        _section("How to Use"),
        ui.tags.ol(
            ui.tags.li("Enter the bounding box coordinates for your study area."),
            ui.tags.li("Choose a cell size (e.g. 0.1° ≈ 11 km at the equator)."),
            ui.tags.li("Select square or hexagonal grid type."),
            ui.tags.li("Optionally fetch a GADM boundary first (Spatial tab) and check 'Clip to GADM'."),
            ui.tags.li("Click 'Generate' to create the grid."),
        ),
        _tip(
            "Start with a coarser cell size to test your workflow quickly, "
            "then reduce cell size for the final analysis. A 0.5° grid over "
            "a country typically gives 500–2,000 PUs; 0.1° gives 10,000+."
        ),
    ),

    # =====================================================================
    # SPATIAL
    # =====================================================================
    "gadm_picker": ui.TagList(
        _section("Overview"),
        ui.p(
            "Fetch administrative boundaries from the Global Administrative "
            "Areas (GADM) database. Boundaries define your study region and "
            "can be used to clip planning unit grids."
        ),
        _section("Parameters"),
        _param_table([
            ("Country", "Select the target country by ISO 3166-1 alpha-3 code (e.g. LTU for Lithuania, USA for United States)."),
            ("Admin Level", "ADM0 = entire country border; ADM1 = state/province; ADM2 = district/county."),
            ("Region Name", "Optional filter to select a specific region within the chosen admin level (e.g. 'California' at ADM1)."),
        ]),
        _section("How to Use"),
        ui.tags.ol(
            ui.tags.li("Select a country from the dropdown."),
            ui.tags.li("Choose the administrative level."),
            ui.tags.li("Optionally enter a region name to filter."),
            ui.tags.li("Click 'Fetch' to download the boundary polygons."),
        ),
        ui.p(
            "The fetched boundary is stored for use by the Grid Builder's "
            "'Clip to GADM' option. It appears as a preview on the map."
        ),
        _tip(
            "Lower admin levels (ADM2) fetch more polygons and may be slower. "
            "Use ADM0 for national analyses and ADM1/ADM2 only when you need "
            "sub-national precision."
        ),
    ),

    # -----------------------------------------------------------------
    "wdpa_overlay": ui.TagList(
        _section("Overview"),
        ui.p(
            "Overlay existing protected areas from the World Database on "
            "Protected Areas (WDPA) onto your planning units. This lets you "
            "lock in PUs that already have legal protection, reflecting the "
            "existing conservation estate in your analysis."
        ),
        _section("Parameters"),
        _param_table([
            ("API Token", "Optional Protected Planet API token for authenticated access. A public endpoint is used if left blank, but may have rate limits."),
            ("Status to Set", "Marxan status to assign overlapping PUs: 'Locked In' (2) — PU must be in the reserve; 'Initial Include' (1) — PU starts selected but can be removed."),
            ("Overlap Threshold", "Minimum fraction (0.1–1.0) of a PU's area that must overlap a protected area to trigger the status change. Default 0.5 = at least 50%."),
        ]),
        _section("How to Use"),
        ui.tags.ol(
            ui.tags.li("Load or create a planning-unit problem with geometry first."),
            ui.tags.li("Optionally enter a Protected Planet API token."),
            ui.tags.li("Choose the status to set (Locked In is most common)."),
            ui.tags.li("Adjust the overlap threshold if needed."),
            ui.tags.li("Click 'Fetch & overlay WDPA' to query and apply."),
        ),
        _tip(
            "Locking in existing protected areas (status=2) is standard practice "
            "in Marxan analyses. It ensures the solver builds on existing "
            "conservation investments rather than ignoring them."
        ),
    ),

    # -----------------------------------------------------------------
    "cost_upload": ui.TagList(
        _section("Overview"),
        ui.p(
            "Upload a custom cost layer to replace or augment the default "
            "planning unit costs. Costs represent what is sacrificed by "
            "including a PU in the reserve — for example land price, "
            "opportunity cost, or acquisition difficulty."
        ),
        _section("Parameters"),
        _param_table([
            ("Cost File", "A vector layer (Shapefile, GeoJSON, GeoPackage) with a numeric cost attribute. Must overlap your planning units spatially."),
            ("Cost Column", "The column in the uploaded file containing the cost values."),
            ("Aggregation", "How to combine costs when a PU overlaps multiple cost polygons: "
             "Area-Weighted Mean (recommended) — weighted average by intersection area; "
             "Sum — total cost; Maximum — highest cost."),
        ]),
        _section("How to Use"),
        ui.tags.ol(
            ui.tags.li("Upload a geospatial file with cost data."),
            ui.tags.li("Select the column containing cost values."),
            ui.tags.li("Choose the aggregation method."),
            ui.tags.li("Click 'Apply Cost' to update planning unit costs."),
        ),
        _section("Marxan Background"),
        ui.p(
            "In Marxan's objective function, cost is the primary component: "
            "Objective = Σ(cost × selected) + BLM × Σ(boundary) + Σ(SPF × shortfall). "
            "Appropriate cost data is critical — using only area as cost ignores "
            "economic and social factors. Common cost proxies include agricultural "
            "land value, timber revenue, and management cost per hectare."
        ),
        _tip(
            "If your cost data has a different CRS than the planning units, "
            "the application reprojects automatically. Ensure both layers "
            "cover the same geographic extent."
        ),
    ),

    # =====================================================================
    # FEATURES
    # =====================================================================
    "feature_table": ui.TagList(
        _section("Overview"),
        ui.p(
            "View and edit conservation feature targets and Species Penalty "
            "Factors (SPF). Features are the biodiversity elements you want "
            "to protect — species, habitat types, ecosystem services, or any "
            "measurable conservation value."
        ),
        _section("Table Columns"),
        _param_table([
            ("ID", "Unique feature identifier (from spec.dat)."),
            ("Name", "Feature name (species, habitat, etc.)."),
            ("Target", "Minimum amount of this feature that must be captured in the reserve. "
             "Units match the amounts in puvspr.dat (e.g. km², proportion, count)."),
            ("SPF", "Species Penalty Factor — controls how strongly Marxan penalises "
             "solutions that fail to meet this feature's target. Higher SPF = stronger "
             "incentive to include enough of this feature. Default is 1.0."),
        ]),
        _section("How to Edit"),
        ui.tags.ol(
            ui.tags.li("Click any cell in the Target or SPF column."),
            ui.tags.li("Type a new value."),
            ui.tags.li("Click 'Save Changes' to apply edits to the active problem."),
        ),
        _section("Marxan Background"),
        ui.p(
            "The SPF penalty in Marxan's objective is: Σ(SPF × shortfall). "
            "If a feature's target is 100 km² and the solution only captures "
            "80 km², the shortfall is 20. With SPF = 10, the penalty is 200. "
            "High-priority features (e.g. critically endangered species) should "
            "have SPF > 1 to ensure the solver prioritises meeting their targets."
        ),
        _tip(
            "Set target = 0 for features with no conservation requirement. "
            "Use the SPF Calibration tool (Calibrate tab) to systematically "
            "find SPF values that ensure all targets are met."
        ),
    ),

    # =====================================================================
    # CONNECTIVITY
    # =====================================================================
    "matrix_input": ui.TagList(
        _section("Overview"),
        ui.p(
            "Upload a connectivity matrix that describes spatial "
            "relationships between planning units. Marxan uses connectivity "
            "in the boundary term of the objective function to promote "
            "spatially cohesive reserve networks."
        ),
        _section("Supported Formats"),
        _param_table([
            ("Edge List", "A CSV file with three columns: id1, id2, value. "
             "Each row is one connection between two planning units. "
             "This is the most common format and is equivalent to bound.dat."),
            ("Full Matrix", "A square N×N CSV where entry (i, j) is the "
             "connectivity weight between PU i and PU j. Suitable for dense "
             "connectivity data."),
        ]),
        _section("How to Use"),
        ui.tags.ol(
            ui.tags.li("Choose the matrix format (Edge List or Full Matrix)."),
            ui.tags.li("Upload your CSV file."),
            ui.tags.li("The application validates the file and displays shape, non-zero count, and density."),
        ),
        _section("Marxan Background"),
        ui.p(
            "The boundary term in Marxan's objective is: BLM × Σ(boundary × selected). "
            "When a pair of adjacent PUs are both selected, their shared boundary "
            "does NOT contribute to the penalty. When only one is selected, the "
            "shared boundary is added. This encourages compact, connected reserves. "
            "The connectivity matrix defines these pairwise relationships."
        ),
        _tip(
            "If you generated a grid or imported a GIS file, boundary data is "
            "computed automatically from polygon adjacency. Upload a custom "
            "matrix only if you need non-standard connectivity (e.g. ecological "
            "corridors, river connections, dispersal kernels)."
        ),
    ),

    # -----------------------------------------------------------------
    "metrics_viz": ui.TagList(
        _section("Overview"),
        ui.p(
            "Compute graph-theoretic metrics from the loaded connectivity "
            "matrix. These metrics help identify critical planning units for "
            "maintaining ecological connectivity."
        ),
        _section("Available Metrics"),
        _param_table([
            ("In-Degree", "Number of incoming connections (column sum). High in-degree PUs are well-connected hubs."),
            ("Out-Degree", "Number of outgoing connections (row sum). For symmetric matrices, in-degree = out-degree."),
            ("Betweenness Centrality", "How often a PU lies on shortest paths between other PU pairs. High betweenness PUs are corridor bottlenecks — removing them fragments the network."),
            ("Eigenvector Centrality", "How connected a PU is to other well-connected PUs. Identifies the most central hubs in the network."),
        ]),
        _section("How to Use"),
        ui.tags.ol(
            ui.tags.li("Load a connectivity matrix first (Matrix Input sub-tab)."),
            ui.tags.li("Click 'Compute Metrics' to calculate all graph metrics."),
            ui.tags.li("Results appear as a table with one row per PU."),
        ),
        _tip(
            "PUs with high betweenness centrality are critical for connectivity "
            "corridors. Consider locking them in (status=2) or increasing their "
            "SPF to ensure they appear in the solution."
        ),
    ),

    # =====================================================================
    # CONFIGURE
    # =====================================================================
    "solver_picker": ui.TagList(
        _section("Overview"),
        ui.p(
            "Select the optimisation algorithm and configure its parameters. "
            "The solver determines how Marxan explores the solution space "
            "and finds reserve designs that minimise cost while meeting "
            "conservation targets."
        ),
        _section("Available Solvers"),
        _param_table([
            ("MIP (PuLP/CBC)", "Mixed Integer Linear Programming. Finds the guaranteed optimal solution. "
             "Best for small–medium problems (<10,000 PUs). Equivalent to R's prioritizr."),
            ("Simulated Annealing (Python)", "Stochastic search using adaptive cooling. Good for large problems. "
             "No external binary needed. Supports multiple independent runs."),
            ("Marxan C++ Binary", "Wraps the original Marxan executable. Well-tested and efficient. "
             "Requires the compiled binary on your system PATH."),
            ("Zone SA", "Simulated annealing for multi-zone problems where each PU is assigned to a management zone."),
            ("Greedy Heuristic", "Selects PUs one-by-one based on a scoring strategy (HEURTYPE). Very fast baseline."),
            ("Iterative Improvement", "Refines an existing solution by trying removals, additions, or swaps (ITIMPTYPE)."),
            ("Pipeline", "Chains heuristic → SA → improvement in Marxan RUNMODE sequences."),
        ]),
        _section("Common Parameters"),
        _param_table([
            ("BLM", "Boundary Length Modifier. Controls the cost-vs-compactness tradeoff. "
             "BLM = 0 ignores boundaries; higher BLM favours compact reserves. "
             "Use BLM Calibration to find the optimal value."),
            ("Number of Solutions", "Independent solver runs. Multiple runs reveal selection frequency "
             "and solution variability. 10–100 for SA; 1 for MIP (deterministic)."),
            ("Random Seed", "For reproducible results. Leave blank or set -1 for random."),
        ]),
        _section("SA Parameters"),
        _param_table([
            ("SA Iterations (NUMITNS)", "Total iterations per run. More = better exploration. Typical: 1,000,000."),
            ("Temperature Steps (NUMTEMP)", "Number of cooling steps. Controls how gradually the system cools. Typical: 10,000."),
        ]),
        _section("MIP Parameters"),
        _param_table([
            ("Time Limit", "Maximum seconds before returning the best solution found so far."),
            ("Optimality Gap", "0.0 = exact optimal; 0.01 = within 1% of optimal. Relaxing this speeds up large problems."),
            ("Verbose", "Print solver progress to the console for debugging."),
        ]),
        _section("Heuristic Parameters"),
        _param_table([
            ("HEURTYPE", "Greedy strategy: 0=Richness, 1=Greedy cheapest, 2=Max Rarity (default), "
             "3=Best Rarity/Cost, 4=Avg Rarity, 5=Sum Rarity, 6=Product Irreplaceability, 7=Summation Irreplaceability."),
            ("ITIMPTYPE", "Improvement strategy: 0=None, 1=Removal pass, 2=Two-step (remove+add), 3=Swap."),
            ("RUNMODE", "Pipeline: 0=SA only, 1=Heuristic only, 2=SA+improvement, 3=Heuristic+improvement, "
             "4=Heuristic+SA, 5=Heuristic+SA+improvement (best quality), 6=Improvement only."),
        ]),
        _tip(
            "Start with MIP for small problems — it gives the provably optimal "
            "solution. Switch to SA for >10,000 PUs where MIP becomes slow."
        ),
    ),

    # -----------------------------------------------------------------
    "zone_config": ui.TagList(
        _section("Overview"),
        ui.p(
            "View the zone definitions and costs for multi-zone Marxan "
            "problems (Marxan with Zones). In standard Marxan, each PU is "
            "either selected or not. In multi-zone planning, each PU is "
            "assigned to exactly one management zone."
        ),
        _section("Typical Zones"),
        _ul(
            "No-take zone — full protection, no resource extraction",
            "Buffer zone — limited activities allowed",
            "Sustainable use zone — managed resource extraction permitted",
            "Available — no special management (the default 'not selected' zone)",
        ),
        _section("Zone Data"),
        ui.p(
            "Zone definitions are loaded from the project files when you "
            "upload a Marxan with Zones project. The table shows each zone "
            "name and its average cost. Zone-specific targets can also be "
            "set in the spec.dat file."
        ),
        _tip(
            "Multi-zone planning is appropriate when you need to balance "
            "multiple land-use objectives, not just 'protect or not'."
        ),
    ),

    # =====================================================================
    # CALIBRATE
    # =====================================================================
    "blm_explorer": ui.TagList(
        _section("Overview"),
        ui.p(
            "Calibrate the Boundary Length Modifier (BLM) to find the best "
            "tradeoff between reserve cost and spatial compactness. The BLM "
            "is arguably the most important parameter in Marxan."
        ),
        _section("The Classic BLM Calibration Approach"),
        ui.tags.ol(
            ui.tags.li("Define a range of BLM values to test (e.g. 0–100)."),
            ui.tags.li("The solver runs once at each BLM value."),
            ui.tags.li("A plot of cost vs. boundary length is produced."),
            ui.tags.li("Look for the 'elbow point' — where increasing BLM starts "
                       "yielding diminishing compactness gains."),
        ),
        _section("Parameters"),
        _param_table([
            ("Min BLM", "Lower bound of the BLM range to explore. Usually 0."),
            ("Max BLM", "Upper bound. Start with 10×cost/boundary ratio; expand if no elbow is visible."),
            ("Steps", "Number of BLM values to test. More steps = smoother curve but slower. 10–20 is usually sufficient."),
        ]),
        _section("Interpreting Results"),
        _ul(
            "Left side of the curve: low BLM, low cost, fragmented reserves",
            "Right side: high BLM, higher cost, very compact reserves",
            "The elbow represents the best cost-efficiency for compactness",
            "If no elbow: try a wider BLM range or check that boundary data is loaded",
        ),
        _section("Marxan Background"),
        ui.p(
            "The BLM multiplies the boundary penalty: BLM × Σ(boundary × selected). "
            "At BLM = 0, the solver ignores spatial arrangement entirely. "
            "At very high BLM, the solver strongly favours compact reserves even at "
            "much higher cost. The 'right' BLM depends on your study system and "
            "conservation objectives."
        ),
        _tip(
            "If the curve is always flat, your boundary data may be zeros or "
            "missing. Check the Connectivity > Matrix Input tab."
        ),
    ),

    # -----------------------------------------------------------------
    "sensitivity": ui.TagList(
        _section("Overview"),
        ui.p(
            "Test how sensitive the reserve design is to changes in "
            "conservation targets. This reveals which features are driving "
            "the solution and how robust the design is to target uncertainty."
        ),
        _section("How It Works"),
        ui.tags.ol(
            ui.tags.li("A range of multipliers is applied to each feature's target "
                       "(e.g. 0.8–1.2 = 80–120% of original)."),
            ui.tags.li("The solver runs once per multiplier value per feature."),
            ui.tags.li("For each run, the objective value and cost are recorded."),
        ),
        _section("Parameters"),
        _param_table([
            ("Multiplier Range", "The min and max multiplier to apply to targets (e.g. 0.5–1.5)."),
            ("Steps", "Number of multiplier values to test within the range."),
        ]),
        _section("Interpreting Results"),
        _ul(
            "Features with steep objective curves are DRIVING the solution — small target changes cause large cost changes",
            "Flat curves indicate the feature is easily met regardless of the exact target",
            "This helps prioritise data collection: focus on accurate targets for the most sensitive features",
        ),
        _tip(
            "Run sensitivity analysis after BLM calibration and with a "
            "reasonable number of solutions to get stable results."
        ),
    ),

    # -----------------------------------------------------------------
    "spf_explorer": ui.TagList(
        _section("Overview"),
        ui.p(
            "Automatically tune Species Penalty Factor (SPF) values to "
            "ensure all conservation targets are met. SPF calibration is "
            "an iterative process that increases penalties for features "
            "that the solver consistently fails to meet."
        ),
        _section("How It Works"),
        ui.tags.ol(
            ui.tags.li("Run the solver with current SPF values."),
            ui.tags.li("Identify features whose targets are NOT met."),
            ui.tags.li("Multiply those features' SPF by the calibration multiplier."),
            ui.tags.li("Repeat until all targets are met or the iteration limit is reached."),
        ),
        _section("Parameters"),
        _param_table([
            ("Max Iterations", "Maximum number of calibration rounds. The process stops early if all targets are met."),
            ("SPF Multiplier", "Factor to increase unmet features' SPF each round. "
             "E.g. 2.0 = double the SPF each iteration. Higher multiplier = faster convergence but may overshoot."),
        ]),
        _section("Marxan Background"),
        ui.p(
            "In Marxan's objective function, SPF × shortfall penalises unmet targets. "
            "If a feature's target cannot be met at SPF = 1, increasing SPF makes it "
            "more 'expensive' for the solver to ignore that feature, eventually forcing "
            "it to include enough PUs. This is the standard Marxan calibration workflow."
        ),
        _tip(
            "Start with a multiplier of 2.0 and max iterations of 10. If the "
            "process doesn't converge, the targets may be infeasible — check that "
            "enough PUs contain the missing features."
        ),
    ),

    # -----------------------------------------------------------------
    "sweep_explorer": ui.TagList(
        _section("Overview"),
        ui.p(
            "Run a systematic sweep over any Marxan parameter to understand "
            "its effect on solution quality. The solver is executed once for "
            "each value in the parameter range."
        ),
        _section("Parameters"),
        _param_table([
            ("Sweep Parameter", "Which parameter to vary: BLM, NUMITNS, or NUMTEMP."),
            ("Min / Max", "Range of values to sweep."),
            ("Steps", "Number of values to test within the range (linearly spaced)."),
        ]),
        _section("How to Use"),
        ui.tags.ol(
            ui.tags.li("Select the parameter to sweep."),
            ui.tags.li("Set the range and number of steps."),
            ui.tags.li("Click 'Run Sweep'."),
            ui.tags.li("Results appear in a table with cost, boundary, and objective per step."),
        ),
        _tip(
            "NUMITNS sweep: useful to find the minimum iterations that give "
            "stable results. If the objective keeps improving at the highest "
            "NUMITNS, you need more iterations."
        ),
    ),

    # =====================================================================
    # RUN
    # =====================================================================
    "run_panel": ui.TagList(
        _section("Overview"),
        ui.p(
            "Execute the configured solver on the loaded conservation "
            "problem. This is where the actual optimisation happens."
        ),
        _section("Before Running"),
        _ul(
            "Step 1: Load a project (Data Input tab) — a planning-unit problem must be active",
            "Step 2: Configure the solver (Configure tab) — choose algorithm and parameters",
            "Step 3: Optionally calibrate BLM and SPF (Calibrate tab)",
        ),
        _section("How to Use"),
        ui.tags.ol(
            ui.tags.li("Click 'Run Solver' to start."),
            ui.tags.li("A progress bar shows completion percentage."),
            ui.tags.li("Status messages update in real time."),
            ui.tags.li("When complete, results are automatically available in Results and Maps tabs."),
        ),
        ui.p(
            "The solver runs in a background thread so the interface "
            "remains responsive. You can navigate to other tabs while "
            "the solver is running."
        ),
        _section("Status Guide"),
        _ul(
            "'Step 1: Go to Data tab and load a Marxan project' — no project loaded yet",
            "'Step 2: Configure solver, then click Run Solver' — project loaded, ready to run",
            "Progress percentage — solver is running",
            "Solution summary — run complete, showing cost/boundary/objective/targets",
        ),
        _tip(
            "If the solver seems stuck, try reducing NUMITNS or switching to "
            "MIP for a quick exact solution on smaller problems."
        ),
    ),

    # =====================================================================
    # MAPS
    # =====================================================================
    "spatial_grid": ui.TagList(
        _section("Overview"),
        ui.p(
            "Interactive map of all planning units colored by cost or "
            "status. Use this to visually inspect the spatial layout of "
            "your conservation planning region before running the solver."
        ),
        _section("Color Modes"),
        _param_table([
            ("Cost", "Yellow (low cost) to red (high cost) gradient. "
             "Helps identify expensive vs. cheap areas for conservation."),
            ("Status", "Categorical colors — gray = available (0/1), "
             "green = locked in (2), red = locked out (3). "
             "Verify that lock-in/lock-out assignments are correct."),
        ]),
        _section("What to Look For"),
        _ul(
            "Spatial distribution of costs — are high-cost PUs clustered?",
            "Locked-in PUs (green) — do existing protected areas cover the right places?",
            "Locked-out PUs (red) — are exclusion zones set correctly?",
            "Gaps or holes — are there PUs missing from your study area?",
        ),
        _tip(
            "If no map appears, ensure your project has geometry (generated "
            "grid or imported GIS) and that ipyleaflet is installed."
        ),
    ),

    # -----------------------------------------------------------------
    "solution_map": ui.TagList(
        _section("Overview"),
        ui.p(
            "Visualise the solver result on an interactive map. Green "
            "planning units are selected for the reserve network; gray "
            "units are not selected."
        ),
        _section("Displayed Information"),
        _ul(
            "Green PUs — selected for the reserve in this solution",
            "Gray PUs — not selected",
            "Summary statistics: number selected, total cost, boundary length, "
            "objective value, and number of targets met",
        ),
        _section("How to Use"),
        ui.tags.ol(
            ui.tags.li("Run the solver first (Run tab)."),
            ui.tags.li("Navigate to Maps > Solution to see the result."),
            ui.tags.li("Zoom and pan to inspect specific areas."),
        ),
        _tip(
            "If multiple solutions were generated, this shows the best "
            "solution (lowest objective value). Compare solutions on the "
            "Comparison sub-tab."
        ),
    ),

    # -----------------------------------------------------------------
    "frequency_map": ui.TagList(
        _section("Overview"),
        ui.p(
            "Heatmap showing how often each planning unit is selected "
            "across multiple solver runs. This is one of the most "
            "important outputs in a Marxan analysis."
        ),
        _section("What It Shows"),
        ui.p(
            "White = never selected (0%) to dark blue = always selected "
            "(100%). Planning units selected in nearly all runs are "
            "considered highly irreplaceable — they are consistently needed "
            "to meet conservation targets."
        ),
        _section("Why It Matters"),
        _ul(
            "High-frequency PUs (>80%) — critical for the reserve network; hard to replace",
            "Medium-frequency PUs (20–80%) — important but with substitutes available",
            "Low-frequency PUs (<20%) — many alternatives exist for these PUs",
            "Selection frequency is a robust measure of conservation priority "
            "even when individual solutions differ",
        ),
        _section("Prerequisites"),
        ui.p(
            "Set 'Number of solutions' > 1 in the Configure tab (typically "
            "10–100) and run the solver. With only one solution, all PUs "
            "are either 0% or 100%."
        ),
        _tip(
            "For publication-quality frequency maps, run 100+ solutions. "
            "The frequency pattern stabilises around 50–100 runs for most "
            "problems."
        ),
    ),

    # -----------------------------------------------------------------
    "comparison_map": ui.TagList(
        _section("Overview"),
        ui.p(
            "Compare two solutions side-by-side to identify shared and "
            "unique planning units. Useful for understanding solution "
            "variability and the effect of different parameter settings."
        ),
        _section("Color Legend"),
        _param_table([
            ("Green", "Selected in BOTH solutions — core areas that appear regardless of the specific run."),
            ("Blue", "Selected in Solution A only — areas unique to the first solution."),
            ("Orange", "Selected in Solution B only — areas unique to the second solution."),
            ("Gray", "Not selected in either — areas excluded from both solutions."),
        ]),
        _section("How to Use"),
        ui.tags.ol(
            ui.tags.li("Run the solver with multiple solutions."),
            ui.tags.li("Select Solution A and Solution B from the dropdowns."),
            ui.tags.li("The map updates to show the four-color comparison."),
        ),
        _tip(
            "Green areas (shared between solutions) are the most robust. "
            "If most PUs are green, the solution is very consistent. If "
            "most are blue/orange, the solutions differ substantially."
        ),
    ),

    # -----------------------------------------------------------------
    "network_view": ui.TagList(
        _section("Overview"),
        ui.p(
            "Visualise the connectivity network overlaid on planning units. "
            "Nodes are colored by a graph metric; edges show connections "
            "above a weight threshold."
        ),
        _section("Parameters"),
        _param_table([
            ("Metric", "Which graph metric to use for node coloring: in-degree, out-degree, betweenness, or eigenvector centrality."),
            ("Edge Threshold", "Minimum connection weight to display an edge. Increase to show only the strongest connections and reduce visual clutter."),
        ]),
        _section("What to Look For"),
        _ul(
            "Dark-colored nodes — high metric value, important for connectivity",
            "Dense edge clusters — well-connected sub-regions",
            "Bridge nodes (high betweenness) — removing these fragments the network",
            "A maximum of 5,000 edges are displayed to prevent browser slowdown",
        ),
        _tip(
            "Load a connectivity matrix first (Connectivity > Matrix Input). "
            "The network view requires both PU geometry and a connectivity "
            "matrix to function."
        ),
    ),

    # =====================================================================
    # RESULTS
    # =====================================================================
    "summary_table": ui.TagList(
        _section("Overview"),
        ui.p(
            "Summary table showing whether each conservation feature's "
            "target is met by the current solution. This is the primary "
            "way to assess solution quality."
        ),
        _section("Table Columns"),
        _param_table([
            ("Feature", "Feature ID and name."),
            ("Target", "The minimum amount required (from spec.dat)."),
            ("Achieved", "The amount actually captured by the selected PUs."),
            ("Percentage", "Achieved ÷ Target × 100%."),
            ("Status", "✓ Met (green) or ✗ NOT MET (red)."),
        ]),
        ui.p(
            "A summary line at the bottom shows 'X of Y targets met'. "
            "Ideally all targets should be met (100%)."
        ),
        _section("If Targets Are Not Met"),
        _ul(
            "Increase SPF for the unmet features (or use SPF Calibration)",
            "Check that the target is achievable — if total feature amount "
            "across all PUs is less than the target, it's infeasible",
            "Increase NUMITNS for SA to allow more exploration",
            "Try the MIP solver for a guaranteed optimal solution",
        ),
        _tip(
            "Use the Target Achievement sub-tab for a sortable/filterable "
            "version of this table."
        ),
    ),

    # -----------------------------------------------------------------
    "target_met": ui.TagList(
        _section("Overview"),
        ui.p(
            "Sortable, filterable data table of conservation targets and "
            "whether each feature is adequately represented in the current "
            "solution. Shows the same information as the Summary Table but "
            "in an interactive DataTable format."
        ),
        _section("How to Use"),
        _ul(
            "Click column headers to sort (e.g. sort by '% Met' to find the least-represented features)",
            "Use the search box to filter by feature name",
            "Columns: Feature ID, Name, Target, Achieved, % Met, Status",
        ),
        _tip(
            "Sort by '% Met' ascending to quickly identify the features "
            "that are hardest to meet. These may need higher SPF values "
            "or revised targets."
        ),
        _section("PROBMODE 3 columns"),
        ui.p(
            "When the active problem has PROBMODE=3 and the current "
            "Solution carries a Z-score evaluation, three additional "
            "columns appear:"
        ),
        _param_table([
            ("ptarget", "Required probability of meeting the deterministic target (set on spec.dat)."),
            ("P(met)", "Actual probability the reserve meets the target, computed from per-cell Bernoulli variance via the Marxan Z-score formulation (Game 2008, Tulloch 2013)."),
            ("prob_gap", "ptarget − P(met), clamped at 0. Non-zero means the reserve falls short of the chance constraint; the solver objective penalises this."),
        ]),
        _tip(
            "If MIP was the solver, ptarget / P(met) / prob_gap are "
            "computed post-hoc on the deterministic solution — the MIP "
            "doesn't optimise against them directly. For chance-constraint "
            "optimality, use SA, heuristic, or iterative-improvement."
        ),
        _section("TARGET2 / clumping columns (Phase 19)"),
        ui.p(
            "When any feature has ``target2 > 0`` (Marxan \"type-4 species\" / "
            "minimum-patch-size constraints), three additional columns appear:"
        ),
        _param_table([
            ("target2", "Minimum amount of the feature required within a single contiguous clump for the clump to count toward the deterministic target."),
            ("clumptype", "How sub-target clumps are scored: 0=binary (no credit), 1=half (occ/2), 2=quadratic (occ²/target2). Verified line-for-line against Marxan v4 clumping.cpp::PartialPen4."),
            ("clump_short", "Raw shortfall: max(0, target·MISSLEVEL − held_eff), where held_eff applies the CLUMPTYPE rule per component. Non-zero means the reserve fails the patch-size requirement; the solver objective penalises this."),
        ]),
        _tip(
            "If MIP was the solver, clump_short is computed post-hoc on the "
            "deterministic solution — the MIP \"drop\" strategy doesn't "
            "optimise against TARGET2 directly. For clumping-aware "
            "optimality, use SA or iterative-improvement. The greedy "
            "heuristic also stays clumping-blind during scoring and only "
            "reports the gap post-hoc."
        ),
        ui.tags.p(
            "References: Ball, Possingham & Watts (2009), Spatial Conservation "
            "Prioritization, Oxford University Press; Metcalfe et al. (2015), "
            "Conservation Biology 29(6): 1615–1625.",
            class_="text-muted small",
        ),
    ),

    # -----------------------------------------------------------------
    "convergence": ui.TagList(
        _section("Overview"),
        ui.p(
            "Plot the simulated annealing convergence curve for each solver "
            "run. This diagnostic tells you whether the solver ran long "
            "enough to find a good solution."
        ),
        _section("Plot Elements"),
        _param_table([
            ("Blue line", "Current objective value at each iteration — should decline and flatten."),
            ("Green line", "Best objective found so far — monotonically decreasing."),
            ("Red dotted line", "Temperature schedule (optional) — shown on a log-scale secondary axis."),
        ]),
        _section("What to Look For"),
        _ul(
            "GOOD: Both lines converge and flatten well before the last iteration — the solver had enough time",
            "BAD: Lines are still declining at the end — increase NUMITNS for better results",
            "Temperature should decrease smoothly from hot to cold — erratic temperature suggests NUMTEMP is too low",
        ),
        _section("Parameters"),
        _param_table([
            ("Run Select", "Choose which solver run to display (if multiple solutions were generated)."),
            ("Show Temperature", "Overlay the temperature curve on the plot."),
        ]),
        _tip(
            "If the convergence curve hasn't flattened after 1,000,000 "
            "iterations, try doubling NUMITNS. For very large problems "
            "(>50,000 PUs), you may need 10,000,000+ iterations."
        ),
    ),

    # -----------------------------------------------------------------
    "scenario_compare": ui.TagList(
        _section("Overview"),
        ui.p(
            "Save the current solution as a named scenario and compare "
            "multiple scenarios side-by-side. This is essential for "
            "evaluating different parameter settings, solver choices, "
            "or target configurations."
        ),
        _section("How to Use"),
        ui.tags.ol(
            ui.tags.li("Run the solver with your first configuration."),
            ui.tags.li("Enter a descriptive name (e.g. 'BLM=10, SA, 100 runs')."),
            ui.tags.li("Click 'Save Current Solution' to store the scenario."),
            ui.tags.li("Change parameters and run again."),
            ui.tags.li("Save with a new name."),
            ui.tags.li("The comparison table shows all saved scenarios with their metrics."),
        ),
        _section("Recorded Metrics"),
        _ul(
            "Total cost of the reserve network",
            "Total boundary length",
            "Objective function value",
            "Number/percentage of targets met",
            "Solver type and key parameters used",
        ),
        _tip(
            "Compare at least 3 scenarios with different BLM values to "
            "present stakeholders with low/medium/high compactness options."
        ),
    ),

    # -----------------------------------------------------------------
    "export": ui.TagList(
        _section("Overview"),
        ui.p(
            "Download solver results as CSV files for use in GIS software, "
            "reports, or further analysis."
        ),
        _section("Available Downloads"),
        _param_table([
            ("Solution CSV", "One row per planning unit with columns: PU ID, selected (1 = in reserve, 0 = not), cost. "
             "Import this into a GIS to visualise the reserve design on your own maps."),
            ("Target Summary CSV", "One row per feature with columns: Feature ID, Name, Target, Achieved amount, Met status. "
             "Use this for reports and tables in publications."),
        ]),
        _section("How to Use"),
        ui.tags.ol(
            ui.tags.li("Run the solver to generate results."),
            ui.tags.li("Click the download button for the CSV you need."),
            ui.tags.li("The file is downloaded to your browser's default download location."),
        ),
        _tip(
            "For GIS mapping, join the Solution CSV to your planning unit "
            "shapefile using the PU ID column. This lets you produce custom "
            "maps in QGIS, ArcGIS, or R."
        ),
    ),
}
