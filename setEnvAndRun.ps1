param(
  [ValidateSet("single","grid","full-grid")]
  [string]$Mode = "single",

  # Spark cluster
    [string]$MasterUrl = "spark://localhost:7077",
  # Experiment params
  [int]$Partitions = 6,
  [int]$Layers = 2,
  [ValidateSet("mlp","rnn","lstm")]
  [string]$ModelType = "mlp",
  [string]$DriverMem = "6g",
  [string]$ExecMem = "4g",
  [int]$ExecCores = 1,
  [int]$NumExec = 2,

  # Optional overrides (recommended for GitHub users)
  [string]$SparkHome = $env:SPARK_HOME,
  [string]$JavaHome  = $env:JAVA_HOME,
  [string]$PyPath    = $env:PYSPARK_PYTHON
)

$ErrorActionPreference = "Stop"

# --- Resolve repo root automatically (repo = parent of /scripts) ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir

# Python script relative to repo root
$PyScript = Join-Path $RepoRoot "05_dl_spark_mlp.py"

function Fail($msg) {
  Write-Host "ERROR: $msg" -ForegroundColor Red
  exit 1
}

function Resolve-IfEmpty {
  param([string]$Value, [string]$Name)
  if ([string]::IsNullOrWhiteSpace($Value)) { return $null }
  return $Value
}

# --- Try to infer SPARK_HOME if missing (common install locations) ---
if ([string]::IsNullOrWhiteSpace($SparkHome)) {
  $candidates = @(
    "C:\spark\spark-3.5.7-bin-hadoop3",
    "C:\spark\spark-3.5.6-bin-hadoop3",
    "C:\spark\spark-3.5.5-bin-hadoop3"
  )
  foreach ($c in $candidates) {
    if (Test-Path $c) { $SparkHome = $c; break }
  }
}

# --- Validate critical files ---
if (-not (Test-Path $PyScript)) {
  Fail "Cannot find $PyScript. Put 05_dl_spark_mlp.py in the repo root or update `$PyScript`."
}

if ([string]::IsNullOrWhiteSpace($SparkHome)) {
  Fail "SPARK_HOME not set and Spark not auto-detected. Pass -SparkHome 'C:\path\to\spark' or set SPARK_HOME."
}

$sparkSubmit = Join-Path $SparkHome "bin\spark-submit.cmd"
if (-not (Test-Path $sparkSubmit)) {
  Fail "spark-submit not found at: $sparkSubmit. Check -SparkHome / SPARK_HOME."
}

# --- Java: use param or env, else try to infer ---
if ([string]::IsNullOrWhiteSpace($JavaHome)) {
  $javaCandidates = @(
    "C:\Program Files\Eclipse Adoptium\jdk-11.0.28.6-hotspot",
    "C:\Program Files\Eclipse Adoptium\jdk-17*",
    "C:\Program Files\Java\jdk*"
  )
  foreach ($jc in $javaCandidates) {
    $hit = Get-Item $jc -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($hit) { $JavaHome = $hit.FullName; break }
  }
}

if ([string]::IsNullOrWhiteSpace($JavaHome)) {
  Fail "JAVA_HOME not set and Java not auto-detected. Pass -JavaHome or set JAVA_HOME."
}

$javaExe = Join-Path $JavaHome "bin\java.exe"
if (-not (Test-Path $javaExe)) {
  Fail "java.exe not found at: $javaExe. Check -JavaHome / JAVA_HOME."
}

# --- Python: must be set, or pass explicitly ---
if ([string]::IsNullOrWhiteSpace($PyPath)) {
  Fail "PYSPARK_PYTHON not set. Pass -PyPath (path to your venv python.exe) or set PYSPARK_PYTHON."
}

if (-not (Test-Path $PyPath)) {
  Fail "Python not found at: $PyPath. Check -PyPath."
}

# --- Export env vars so Spark uses your Python/Java ---
$env:JAVA_HOME = $JavaHome
$env:SPARK_HOME = $SparkHome
$env:PYSPARK_PYTHON = $PyPath
$env:PYSPARK_DRIVER_PYTHON = $PyPath

# Add spark/java to PATH for this session
$env:PATH = "$env:JAVA_HOME\bin;$env:SPARK_HOME\bin;$env:SPARK_HOME\sbin;$env:PATH"

Write-Host "RepoRoot:  $RepoRoot"
Write-Host "SPARK_HOME:$env:SPARK_HOME"
Write-Host "JAVA_HOME: $env:JAVA_HOME"
Write-Host "PYSPARK:   $env:PYSPARK_PYTHON"
Write-Host "Script:    $PyScript"
Write-Host ""

$script:sparkProcs = @()

function Stop-SparkCluster {
  Write-Host "Stopping Spark cluster..." -ForegroundColor Yellow
  foreach ($p in $script:sparkProcs) {
    if (-not $p.HasExited) {
      # Kill the cmd.exe wrapper and its Java child
      $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)" -ErrorAction SilentlyContinue
      foreach ($c in $children) { Stop-Process -Id $c.ProcessId -Force -ErrorAction SilentlyContinue }
      Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
  }
  $script:sparkProcs = @()
  Start-Sleep -Seconds 3
}

function Start-SparkCluster {
  param([int]$NumWorkers, [int]$CoresPerWorker, [string]$MemPerWorker)
  $sparkClass = Join-Path $SparkHome "bin\spark-class.cmd"

  Write-Host "Starting master..." -ForegroundColor Yellow
  $p = Start-Process -FilePath $sparkClass `
    -ArgumentList "org.apache.spark.deploy.master.Master --host localhost --port 7077 --webui-port 8080" `
    -PassThru -WindowStyle Minimized
  $script:sparkProcs += $p
  Start-Sleep -Seconds 5

  Write-Host "Starting $NumWorkers worker(s) - ${CoresPerWorker} cores / ${MemPerWorker} each..." -ForegroundColor Yellow
  for ($i = 0; $i -lt $NumWorkers; $i++) {
    $p = Start-Process -FilePath $sparkClass `
      -ArgumentList "org.apache.spark.deploy.worker.Worker spark://localhost:7077 --cores $CoresPerWorker --memory $MemPerWorker" `
      -PassThru -WindowStyle Minimized
    $script:sparkProcs += $p
    Start-Sleep -Seconds 2
  }
  Start-Sleep -Seconds 3
  Write-Host "Cluster ready. Check http://localhost:8080" -ForegroundColor Green
}

function Run-One {
  param(
    [int]$P,
    [int]$Layers,
    [string]$Model = "mlp",
    [string]$Dmem,
    [string]$Emem,
    [int]$Cores,
    [int]$Execs
  )

  Write-Host "RUN => model=$Model partitions=$P layers=$Layers driverMem=$Dmem execMem=$Emem execCores=$Cores execInstances=$Execs"

  & $sparkSubmit `
    --master $MasterUrl `
    --deploy-mode client `
    --name "DLS-IDS-$Model" `
    --driver-memory $Dmem `
    --conf "spark.executor.memory=$Emem" `
    --conf "spark.executor.cores=$Cores" `
    --conf "spark.executor.instances=$Execs" `
    $PyScript $P $Layers $Model
}

if ($Mode -eq "single") {
  Run-One -P $Partitions -Layers $Layers -Model $ModelType -Dmem $DriverMem -Emem $ExecMem -Cores $ExecCores -Execs $NumExec
}
elseif ($Mode -eq "full-grid") {
  # full-grid: manages the cluster automatically, restarting workers between config groups
  # Group A & C share the same worker setup (2 cores / 7g); Group B needs 4 cores / 15g

  $groups = @(
    @{ Label="A_2w_2c_7g_p4";  Workers=4; Cores=2; Mem="7g";  Partitions=4; Execs=2 },
    @{ Label="B_2w_4c_15g_p8"; Workers=2; Cores=4; Mem="15g"; Partitions=8; Execs=2 },
    @{ Label="C_4w_2c_7g_p8";  Workers=4; Cores=2; Mem="7g";  Partitions=8; Execs=4 }
  )

  foreach ($grp in $groups) {
    Stop-SparkCluster
    Start-SparkCluster -NumWorkers $grp.Workers -CoresPerWorker $grp.Cores -MemPerWorker $grp.Mem

    foreach ($M in @("mlp","rnn","lstm")) {
      foreach ($L in @(1,2,3)) {
        Write-Host ""
        Write-Host "==== CONFIG $($grp.Label) | model=$M | layers=$L ====" -ForegroundColor Cyan
        Run-One -P $grp.Partitions -Layers $L -Model $M -Dmem $DriverMem -Emem $grp.Mem -Cores $grp.Cores -Execs $grp.Execs
      }
    }
  }

  Stop-SparkCluster
  Write-Host "full-grid complete." -ForegroundColor Green
}
else {
  $configs = @(
    # A) 2 workers, 2 cores/worker, exec mem 7g, partitions 4
    @{ Name="A_2w_2c_7g_p4";  Execs=2; Cores=2; ExecMem="7g";  Partitions=4 },

    # B) 2 workers, 4 cores/worker, exec mem 15g, partitions 8
    @{ Name="B_2w_4c_15g_p8"; Execs=2; Cores=4; ExecMem="15g"; Partitions=8 },

    # C) 4 workers, 2 cores/worker, exec mem 7g, partitions 8
    @{ Name="C_4w_2c_7g_p8";  Execs=4; Cores=2; ExecMem="7g";  Partitions=8 }
  )

  $layersList = @(1,2,3)

  foreach ($cfg in $configs) {
    foreach ($M in @("mlp","rnn","lstm")) {
      foreach ($L in $layersList) {
        Write-Host ""
        Write-Host "==== CONFIG $($cfg.Name) | model=$M | layers=$L ====" -ForegroundColor Cyan

        Run-One `
          -P      $cfg.Partitions `
          -Layers $L `
          -Model  $M `
          -Dmem   $DriverMem `
          -Emem   $cfg.ExecMem `
          -Cores  $cfg.Cores `
          -Execs  $cfg.Execs
      }
    }
  }
}