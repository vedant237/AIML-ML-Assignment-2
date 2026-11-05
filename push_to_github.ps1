# Push from C:\AIML-ML-Assignment-2 to https://github.com/vedant237/AIML-ML-Assignment-2
# Prerequisites: Git installed, run once: gh auth login

$ErrorActionPreference = "Stop"
Set-Location "C:\AIML-ML-Assignment-2"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Install Git: https://git-scm.com/download/win"
    exit 1
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Host "Install GitHub CLI: https://cli.github.com/"
    exit 1
}

gh auth status 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Run: gh auth login"
    exit 1
}

if (-not (Test-Path .git)) {
    git init
    git branch -M main
}

git add app.py train_models.py requirements.txt README.md .gitignore .streamlit test_data.csv mobile_price_dataset.csv training_metrics.json model/
git status

$changes = git status --porcelain
if ($changes) {
    git commit -m "AIML ML Assignment 2: mobile price classification and Streamlit app"
}

$remote = git remote get-url origin 2>$null
if (-not $remote) {
    gh repo create vedant237/AIML-ML-Assignment-2 --public --source=. --remote=origin --push
} else {
    git push -u origin main
}

Write-Host "Done: https://github.com/vedant237/AIML-ML-Assignment-2"
