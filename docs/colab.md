# Running on Google Colab

There are three routes. They share one rule: **a Colab VM bills until it is stopped**.

| Route | Best for | Results | Cleanup |
|---|---|---|---|
| [`notebooks/colab.ipynb`](../notebooks/colab.ipynb) in the browser | exploring, teaching | download cell / Drive | close or disconnect the runtime |
| `quantum-workflow-colab` from your terminal | repeatable jobs, agents, scripts | extracted into `--output` | automatic `colab stop` |
| raw `colab exec` / `colab run` | ad hoc experiments | manual `colab download` | manual `colab stop` |

The last two use Google's [`colab` CLI](https://pypi.org/project/google-colab-cli/) (checked against
**google-colab-cli 0.7.5**; Python ≥ 3.12, Linux/macOS):

```bash
uv tool install google-colab-cli      # or: pip install google-colab-cli
colab version
```

## 1. Authenticate once

The CLI's default is OAuth (`--auth=oauth2`, not ADC). Run this once in a real terminal and complete the
Google sign-in:

```bash
colab sessions
```

The token is cached in `~/.config/colab-cli/token.json`. Treat that file as a credential: never commit,
print or upload it. From then on, every command (and `quantum-workflow-colab`) works unattended. To check
auth from a script without hanging on the prompt, close stdin: `colab sessions </dev/null` prints a sign-in
URL and exits 1 when you are not logged in.

`colab auth -s NAME` is unrelated: it gives *code running on the VM* GCP credentials. It never fixes a CLI 401/403.

## 2. Run a job with the driver

```bash
quantum-workflow-colab examples/lih-bk.json --output results/lih-colab --dry-run   # show the plan
quantum-workflow-colab examples/lih-bk.json --output results/lih-colab
quantum-workflow-colab examples/h2o-dft-gpu.json --output results/h2o-gpu --gpu T4
```

What it does, in order:

1. Validates the config **locally**, before anything is rented. GPU configs require `--gpu`. Unknown GPU names
   are rejected, because the CLI silently turns them into an A100.
2. Checks auth and that the session name is unused. Reusing a name silently orphans the older VM, which keeps
   billing.
3. Builds a wheel of your checkout, uncommitted changes included (`--source local`, the default). Use
   `--source git+https://github.com/<you>/<repo>@<ref>` to install a pushed ref, or `--source preinstalled`.
4. `colab new` → creates `/content/<session>` → uploads the config, pseudopotentials and wheel.
5. `colab exec -f quantum_workflow/colab_remote.py --timeout 3600`. The remote runner installs the project (plus
   QE via micromamba or GPU4PySCF when the config needs them), runs the job in a subprocess, archives
   `results/` without QE scratch files, and prints `__QW_JOB_OK__` as its last line on success.
6. Checks for that sentinel and downloads and safely extracts the archive into `--output`.
7. `colab stop`, in a `finally`, so errors and Ctrl-C still release the VM. `--keep` skips this and prints the
   stop command instead.

Then run `colab sessions` and make sure nothing is left. A `[?]` line is a VM you are paying for with no local
handle (see Recovery below).

## 3. Traps the driver handles (and you must handle with raw `colab`)

- **Exit codes lie.** `colab exec` exits 0 on uncaught exceptions and on `sys.exit(N)`; a failing notebook
  cell does not stop later cells. Detect success with a sentinel line, not `$?`. `colab run` does propagate
  exit codes.
- **`--timeout` defaults to 30 s** on `exec` and `run`, and it is not a kill switch. Past it the job keeps
  running while the client busy-polls a CPU core. Set it at least as long as the job. For hours-long work,
  detach on the VM (`subprocess.Popen(..., start_new_session=True)`) and poll a log file.
- **Two working directories.** `exec`/`run` start in `/content`, but `upload`/`download`/`ls`/`rm` resolve paths
  from `/`. Always pass `/content/...` to file commands.
- **Uploads** are single files only, and the remote parent directory must already exist (otherwise HTTP 500).
  Directories must be tarred.
- **`.ipynb` downloads arrive corrupted** (a Python dict repr). Copy the notebook to a `.json` name on the VM,
  download that, then rename it.
- **`--env KEY=VALUE` is logged in plaintext** in `~/.config/colab-cli/history/` and in `colab log` exports.
  Never pass secrets that way. Upload a secrets file instead, which logs only its path.
- **Never pipe code into `colab exec` before auth works.** The first line is consumed as the OAuth code.
- **Each call has about 10 s of teardown overhead.** Batch work into fewer calls.

## 4. Raw CLI recipes

One-shot job, no files back (the VM is released afterwards; exit codes propagate):

```bash
colab run --timeout 1800 quantum_workflow/colab_remote.py \
  --job-dir /content/job --repo https://github.com/<you>/<repo>.git --ref main \
  --install repo --extras qubit --config examples/h2o-cas44.json
```

The summary JSON is printed to stdout, and the files are discarded with the VM. Use the driver or `--keep`
plus `colab download` to retrieve them.

A notebook, headless:

```bash
colab new -s nb-job
colab exec -s nb-job -f notebooks/colab.ipynb --timeout 1800    # writes colab_output.ipynb locally
python -c "import json,sys; nb=json.load(open(sys.argv[1])); sys.exit(any(o.get('output_type')=='error' for c in nb['cells'] for o in c.get('outputs',[])))" notebooks/colab_output.ipynb
colab stop -s nb-job
```

## 5. Costs and limits

- `colab usage` shows your compute-unit balance and burn rate. Run it before asking for large GPUs.
- Supported shapes: `--gpu T4|L4|G4|H100|A100`, `--high-mem` (Colab Pro/Pro+). Availability depends on your
  tier; HTTP 400 means no entitlement (fall back to `T4` or CPU), and 412 means too many sessions or no capacity.
- Nothing in this project uses TPUs. Every example here runs on CPU in seconds; GPUs only pay off for larger
  DFT (GPU4PySCF), and qubit export and ADAPT stay on CPU.

## 6. Recovery

- **Orphaned `[?]` VM**: copy its endpoint from `colab sessions`, then
  ```bash
  echo '{"orphan": {"name": "orphan", "token": "", "url": "", "endpoint": "<ENDPOINT>"}}' > /tmp/orphan.json
  colab --config /tmp/orphan.json stop -s orphan </dev/null
  ```
- **"Session not found" / 401 on exec**: the backend pruned the VM. Run `colab sessions`, then create a new one.
- **Wedged kernel**: `colab restart-kernel -s NAME` keeps the VM; otherwise `colab stop` and start again.
- **HTTP debugging**: `colab --logtostderr <cmd>` or `~/.config/colab-cli/colab.log`.

Status: the driver is tested against a fake `colab` CLI that mimics 0.7.5's command shapes and its exit-0-on-
failure behaviour (`tests/fake_colab.py`). It has not yet been run against a live Colab VM from this
repository, so check the first real run with `--dry-run` and then `colab sessions`.
