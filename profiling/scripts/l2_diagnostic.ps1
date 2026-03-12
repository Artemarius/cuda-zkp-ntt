# l2_diagnostic.ps1 -- L2 cache diagnostic for outer kernel (v1.5.0 Session 12, Part A)
#
# Profiles the cooperative outer kernel with Nsight Compute using the built-in
# "memory" section set. Saves .ncu-rep files for ncu-ui inspection.
#
# Usage: powershell -ExecutionPolicy Bypass -File profiling\scripts\l2_diagnostic.ps1
# Requires: ncu in PATH or default install location
# Binary:   build\Release\ntt_profile.exe

$ErrorActionPreference = 'Continue'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path "$ScriptDir\..\.."
$Binary = "$RepoRoot\build\Release\ntt_profile.exe"
$ResultsDir = "$RepoRoot\results\data"

if (-not (Test-Path $Binary)) {
    Write-Host "ERROR: $Binary not found."
    Write-Host 'Build with: cmake --build build --config Release --target ntt_profile'
    exit 1
}

if (-not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null
}

# Find ncu
$ncu = Get-Command ncu -ErrorAction SilentlyContinue
if (-not $ncu) { $ncu = Get-Command ncu.bat -ErrorAction SilentlyContinue }
if (-not $ncu) {
    $ncuPaths = @(
        'C:\Program Files\NVIDIA Corporation\Nsight Compute *\ncu.bat',
        'C:\Program Files\NVIDIA GPU Computing Toolkit\nsight-compute\*\ncu.bat'
    )
    foreach ($pattern in $ncuPaths) {
        $found = Get-Item $pattern -ErrorAction SilentlyContinue | Sort-Object -Descending | Select-Object -First 1
        if ($found) { $ncu = $found; break }
    }
}
if (-not $ncu) {
    Write-Host 'ERROR: ncu not found. Install Nsight Compute or add it to PATH.'
    exit 1
}
$ncuPath = if ($ncu -is [System.Management.Automation.ApplicationInfo]) { $ncu.Source } else { $ncu.FullName }
Write-Host "Using ncu: $ncuPath"
Write-Host ''

$Sizes = @(18, 20, 22)

# ---------------------------------------------------------------------------
# SECTION 1: Outer kernel at each size -- full section set
# ---------------------------------------------------------------------------

Write-Host '------------------------------------------------------------------'
Write-Host ' SECTION 1: Outer kernel -- profile and save .ncu-rep'
Write-Host '------------------------------------------------------------------'
Write-Host ''

foreach ($Size in $Sizes) {
    $N = [int64][math]::Pow(2, $Size)
    $MB = [math]::Floor($N * 32 / 1024 / 1024)
    $ReportFile = "$ResultsDir\l2_diag_outer_2e${Size}"

    Write-Host ">>> 2^$Size ($MB MB data) -- profiling outer kernel..."

    & $ncuPath `
        --set full `
        --kernel-name-base function `
        --kernel-name 'regex:radix[48]' `
        --target-processes all `
        --export $ReportFile `
        --force-overwrite `
        $Binary --mode barrett --size $Size 2>&1 | ForEach-Object {
            $line = $_.ToString()
            if ($line -match 'radix[48]') { Write-Host "  $line" }
        }

    Write-Host ''
}

# ---------------------------------------------------------------------------
# SECTION 2: All kernels at 2^22
# ---------------------------------------------------------------------------

Write-Host '------------------------------------------------------------------'
Write-Host ' SECTION 2: All kernels at 2^22 -- profile and save .ncu-rep'
Write-Host '------------------------------------------------------------------'
Write-Host ''

$ReportFile = "$ResultsDir\l2_diag_all_2e22"
Write-Host '>>> 2^22 -- profiling ALL kernels...'

& $ncuPath `
    --set full `
    --kernel-name-base function `
    --target-processes all `
    --export $ReportFile `
    --force-overwrite `
    $Binary --mode barrett --size 22 2>&1 | ForEach-Object {
        $line = $_.ToString()
        if ($line -match 'kernel|radix|fused|bit_reverse') { Write-Host "  $line" }
    }

Write-Host ''

# ---------------------------------------------------------------------------
# SECTION 3: Extract key metrics from .ncu-rep (one metric at a time)
# Bulk --csv export causes "bad conversion"; single-metric queries work.
# ---------------------------------------------------------------------------

Write-Host '------------------------------------------------------------------'
Write-Host ' SECTION 3: Key L2 metrics extracted from .ncu-rep files'
Write-Host '------------------------------------------------------------------'
Write-Host ''

# Metric name -> display label mapping (ordered)
$MetricTable = [ordered]@{
    'lts__t_sector_hit_rate.pct'                       = 'L2 Hit Rate'
    'dram__bytes_read.sum'                             = 'DRAM Bytes Read'
    'dram__bytes_write.sum'                            = 'DRAM Bytes Written'
    'sm__warps_active.avg.pct_of_peak_sustained_active' = 'Occupancy'
}

# Helper: query a single metric from a .ncu-rep file and return the value(s).
# Parses the CSV output, skipping the header row and any "bad conversion" lines.
# Returns an array of [kernel_short_name, value] pairs.
function Extract-NcuMetric {
    param(
        [string]$Report,
        [string]$Metric
    )
    $results = @()
    $raw = & $ncuPath --import $Report --csv --metrics $Metric 2>&1
    foreach ($line in $raw) {
        $s = $line.ToString()
        # Skip empty lines, header, and bad conversion warnings
        if ($s -match 'bad conversion') { continue }
        if ($s -match '"Metric Name"') { continue }
        if ([string]::IsNullOrWhiteSpace($s)) { continue }
        if ($s -notmatch $Metric) { continue }

        # CSV fields are quoted. Split on '","' after stripping leading/trailing quotes.
        $stripped = $s.TrimStart('"').TrimEnd('"')
        $fields = $stripped -split '","'
        if ($fields.Count -lt 2) { continue }

        $val = $fields[-1].Trim()
        # Kernel name is typically field index 4 (0-based) in ncu CSV
        $kernel = ''
        if ($fields.Count -ge 5) {
            $kernel = $fields[4].Trim()
            # Shorten to just the function name before '('
            if ($kernel -match '(\w+)\(') { $kernel = $Matches[1] }
        }
        $results += ,@($kernel, $val)
    }
    return $results
}

# --- Outer kernel reports (per size) ---
foreach ($Size in $Sizes) {
    $ReportFile = "$ResultsDir\l2_diag_outer_2e${Size}.ncu-rep"
    if (-not (Test-Path $ReportFile)) {
        Write-Host "  WARNING: $ReportFile not found, skipping."
        continue
    }

    Write-Host "--- 2^$Size outer kernel ---"

    foreach ($metricName in $MetricTable.Keys) {
        $label = $MetricTable[$metricName]
        $pairs = Extract-NcuMetric -Report $ReportFile -Metric $metricName
        if ($pairs -and $pairs.Count -gt 0) {
            foreach ($pair in $pairs) {
                if ($pair -is [System.Array] -and $pair.Count -ge 2) {
                    Write-Host "  $label = $($pair[1])"
                }
            }
        } else {
            Write-Host "  $label = (not available)"
        }
    }
    Write-Host ''
}

# --- All-kernels report at 2^22 (per-kernel DRAM traffic) ---
$AllReport = "$ResultsDir\l2_diag_all_2e22.ncu-rep"
if (Test-Path $AllReport) {
    Write-Host '--- All kernels at 2^22 (per-kernel breakdown) ---'

    foreach ($metricName in $MetricTable.Keys) {
        $label = $MetricTable[$metricName]
        $pairs = Extract-NcuMetric -Report $AllReport -Metric $metricName
        if ($pairs -and $pairs.Count -gt 0) {
            foreach ($pair in $pairs) {
                if ($pair -is [System.Array] -and $pair.Count -ge 2) {
                    $kname = $pair[0]
                    $kval  = $pair[1]
                    if ($kname) {
                        Write-Host "  ${kname}: $label = $kval"
                    } else {
                        Write-Host "  $label = $kval"
                    }
                }
            }
        }
    }
    Write-Host ''
}

# ---------------------------------------------------------------------------
# Decision-point summary
# ---------------------------------------------------------------------------

Write-Host '------------------------------------------------------------------'
Write-Host ' DECISION FRAMEWORK'
Write-Host '------------------------------------------------------------------'
Write-Host ''
Write-Host 'Key metric: lts__t_sector_hit_rate.pct at 2^22'
Write-Host ''
Write-Host '  Under 20%  -->  LATENCY-BOUND --> Stockham GO (v1.8.0)'
Write-Host '  20-50%     -->  MIXED         --> Stockham CONDITIONAL'
Write-Host '  Over 50%   -->  BANDWIDTH-BOUND --> Stockham NO-GO'
Write-Host ''
Write-Host '  Working set: 2^18=8MB, 2^20=32MB, 2^22=128MB (L2=3MB)'
Write-Host ''
Write-Host ('.ncu-rep files saved to ' + $ResultsDir)
Write-Host 'Open in ncu-ui for roofline and detailed analysis.'
Write-Host 'Done.'
