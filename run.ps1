# Config
$imageName = "bayespace"
$containerName = "bayespace-container"
$projectPath = "/BayeSpace"
$localPath = (Get-Location).Path
$vscodeDir = "$localPath\.vscode"
$settingsFile = "$vscodeDir\settings.json"

# Ensure folders exist
if (-not (Test-Path "$localPath\data")) { New-Item -ItemType Directory -Path "$localPath\data" | Out-Null }
if (-not (Test-Path "$localPath\results")) { New-Item -ItemType Directory -Path "$localPath\results" | Out-Null }

# Disable restricted mode for this workspace
if (-not (Test-Path $vscodeDir)) {
    New-Item -ItemType Directory -Path $vscodeDir | Out-Null
}
$settingsJson = @"
{
    "security.workspace.trust.enabled": false
}
"@
$settingsJson | Out-File -Encoding utf8 $settingsFile

# Check if container already exists
$existingContainer = docker ps -a --format "{{.Names}}" | Where-Object { $_ -eq $containerName }

if ($existingContainer) {
    $response = Read-Host "A container named '$containerName' already exists. Do you want to delete and recreate it? (Y/N)"
    if ($response -match '^[Yy]$') {
        docker stop $containerName | Out-Null
        docker rm $containerName | Out-Null
        Write-Host "Existing container removed. Rebuilding..."
    } else {
        Write-Host "Aborting setup. Container still running."
        exit
    }
}

# Build the Docker image from GitHub
docker build -t $imageName https://github.com/samcd22/BayeSpace.git

# Run the container (no notebooks mount to preserve container content)
docker run -dit `
  --name $containerName `
  -v "$localPath\data:$projectPath/data" `
  -v "$localPath\results:$projectPath/results" `
  -w $projectPath `
  $imageName

# Wait a moment to ensure container is running
Start-Sleep -Seconds 3

# If notebooks don't already exist locally, copy from container
if (-not (Test-Path "$localPath\notebooks")) {
    docker cp "${containerName}:${projectPath}/notebooks" "$localPath\notebooks"
    Write-Host "Copied notebooks from container to local directory."
} else {
    Write-Host "Local notebooks directory already exists. Skipping copy."
}
