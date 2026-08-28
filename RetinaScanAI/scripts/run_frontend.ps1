# Serves the static frontend on http://localhost:5500
Set-Location "$PSScriptRoot\..\frontend"
python -m http.server 5500
