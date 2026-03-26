# Snakefile
# ─────────────────────────────────────────────────────────────────────────────
# Causality Analysis — end-to-end pipeline
#
# Run from the repo root in the activated snakemake-host environment:
#
#   snakemake --cores 1              # full run  (all rules)
#   snakemake --cores 1 simulate     # data generation only
#   snakemake --cores 1 run_fc       # FC benchmark only
#   snakemake --cores 1 bids_convert # BIDS conversion only
#   snakemake --cores 1 fc_demo      # demo notebook + figures only
#   snakemake --dryrun               # preview without executing
#   snakemake --forceall --cores 1   # re-run everything
#
# DAG:
#   simulate ──► run_fc
#   simulate ──► (data re-generated inside notebook) ──► fc_demo
#   bids_convert   (independent: generates its own 5-subject data internally)
# ─────────────────────────────────────────────────────────────────────────────

configfile: "config.yml"
container: "docker://viola1003/causality-analysis:1.0"

# ── Derived constants ─────────────────────────────────────────────────────────
_PY = "python"

# Figure files produced by fc_random_demo.ipynb
DEMO_FIGS = [
    "fig1_ground_truth",
    "fig2_time_series",
    "fig3_ground_truth_matrix",
    "fig4_fc_matrix_grid",
    "fig4b_mcc_bar",
    "fig5_mcc_bar",
    "fig5_auc_roc_avg_precision",
    "fig5_lag_estimation",
    "fig6_dtf_freq_profile",
    "fig6_pdcoh_freq_profile",
]


# ── Rule: all (default target) ────────────────────────────────────────────────
rule all:
    """Build every pipeline output."""
    input:
        "data/simulated_connectivity_benchmark.npz",
        "data/mcc_benchmark.pkl",
        "data/bids/dataset_description.json",
        expand("figures/{fig}.pdf", fig=DEMO_FIGS),


# ── Rule 1: simulate ─────────────────────────────────────────────────────────
rule simulate:
    """
    Generate simulated multi-channel EEG data for all subjects, models, and
    epochs (scripts/01_simulate.py).

    Output : data/simulated_connectivity_benchmark.npz
             shape = (n_subjects, n_models, n_epochs, n_nodes=5, T)
    """
    output:
        "data/simulated_connectivity_benchmark.npz",
    log:
        "logs/simulate.log",
    params:
        py         = _PY,
        n_subjects = config["n_subjects"],
        n_epochs   = config["n_epochs"],
        t          = config["t"],
        models     = " ".join(config["models"]),
    shell:
        "{params.py} scripts/01_simulate.py"
        " --output     {output}"
        " --n-subjects {params.n_subjects}"
        " --n-epochs   {params.n_epochs}"
        " --t          {params.t}"
        " --models     {params.models}"
        " > {log} 2>&1"


# ── Rule 2: run_fc ────────────────────────────────────────────────────────────
rule run_fc:
    """
    Run all FC methods on the benchmark data and evaluate against ground-truth
    adjacency matrices using MCC (scripts/02_run_fc.py).

    Input  : data/simulated_connectivity_benchmark.npz
    Output : data/mcc_benchmark.pkl
             {model_name: {method_name: [mcc_per_subject]}}
    """
    input:
        "data/simulated_connectivity_benchmark.npz",
    output:
        "data/mcc_benchmark.pkl",
    log:
        "logs/run_fc.log",
    params:
        py         = _PY,
        n_subjects = config["n_subjects"],
        percentile = config["percentile"],
        methods    = " ".join(config["methods"]),
        fs         = config["fs"],
    shell:
        "{params.py} scripts/02_run_fc.py"
        " --data       {input}"
        " --output     {output}"
        " --n-subjects {params.n_subjects}"
        " --percentile {params.percentile}"
        " --methods    {params.methods}"
        " --fs         {params.fs}"
        " > {log} 2>&1"


# ── Rule 3: bids_convert ──────────────────────────────────────────────────────
rule bids_convert:
    """
    Execute notebooks/01_bids_conversion.ipynb and write the executed
    version to logs/01_bids_conversion_executed.ipynb.

    Generates 5 simulated subjects (random model) and writes a BIDS-compliant
    EEG dataset to data/bids/.  Channel names are kept as x1-x5 so it is
    immediately clear that the signals are simulated.

    Output : data/bids/dataset_description.json  (sentinel file)
    """
    output:
        "data/bids/dataset_description.json",
    log:
        "logs/bids_convert.log",
    shell:
        "conda run --no-capture-output -n fc_jupyter "
        "jupyter nbconvert --to notebook --execute "
        "--ExecutePreprocessor.timeout=3600 "
        "--output-dir logs/ "
        "--output 01_bids_conversion_executed "
        "notebooks/01_bids_conversion.ipynb "
        "> {log} 2>&1"
        

# ── Rule 4: fc_demo ───────────────────────────────────────────────────────────
rule fc_demo:
    """
    Execute notebooks/fc_random_demo.ipynb and write the executed
    version to logs/fc_random_demo_executed.ipynb.
    
    Runs all FC methods on the random model and saves publication-quality
    figures to figures/ (PDF + PNG at 300 dpi).

    Output : figures/fig1_ground_truth.pdf  …  figures/fig6_pdcoh_freq_profile.pdf
    """
    output:
        expand("figures/{fig}.pdf", fig=DEMO_FIGS),
    log:
        "logs/fc_demo.log",
    shell:
        "conda run --no-capture-output -n fc_jupyter "
        "jupyter nbconvert --to notebook --execute "
        "--ExecutePreprocessor.timeout=3600 "
        "--output-dir logs/ "
        "--output fc_random_demo_executed "
        "notebooks/fc_random_demo.ipynb "
        "> {log} 2>&1"
