set shell := ["bash", "-uc"]
cache_dir := ".cache"
go_cache := cache_dir + "/go-build"
golangci_cache := cache_dir + "/golangci-lint"

# Default recipe - show available commands
default:
    @just --list

# Format source files.
fmt:
    treefmt --allow-missing-formatter --no-cache --tree-root .

# Verify formatting without modifying files.
check-formatted:
    treefmt --allow-missing-formatter --fail-on-change --no-cache --tree-root .

# Run linters.
lint:
    mkdir -p {{go_cache}} {{golangci_cache}}
    cache_root="$(pwd)/{{cache_dir}}"; GOCACHE="${GOCACHE:-$cache_root/go-build}" GOLANGCI_LINT_CACHE="${GOLANGCI_LINT_CACHE:-$cache_root/golangci-lint}" golangci-lint run --timeout=2m ./...

# Run linters with auto-fix where supported.
lint-fix:
    mkdir -p {{go_cache}} {{golangci_cache}}
    cache_root="$(pwd)/{{cache_dir}}"; GOCACHE="${GOCACHE:-$cache_root/go-build}" GOLANGCI_LINT_CACHE="${GOLANGCI_LINT_CACHE:-$cache_root/golangci-lint}" golangci-lint run --fix --timeout=2m ./...

# Ensure module files are tidy.
check-tidy:
    mkdir -p {{go_cache}}
    cache_root="$(pwd)/{{cache_dir}}"; GOCACHE="${GOCACHE:-$cache_root/go-build}" go mod tidy
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then git diff --exit-code go.mod go.sum; else echo "not a git repo; skipped git diff"; fi

# Run tests.
test:
    mkdir -p {{go_cache}}
    cache_root="$(pwd)/{{cache_dir}}"; GOCACHE="${GOCACHE:-$cache_root/go-build}" go test ./...

# Run tests with coverage.
test-coverage:
    mkdir -p {{go_cache}}
    cache_root="$(pwd)/{{cache_dir}}"; GOCACHE="${GOCACHE:-$cache_root/go-build}" go test -coverprofile=coverage.out ./...
    go tool cover -html=coverage.out -o coverage.html

# Build the CLI tool.
build:
    mkdir -p bin
    mkdir -p {{go_cache}}
    cache_root="$(pwd)/{{cache_dir}}"; GOCACHE="${GOCACHE:-$cache_root/go-build}" go build -o bin/microgpt ./cmd/microgpt

# Build web demo artifacts in web/dist.
web-build:
    mkdir -p web/dist
    mkdir -p {{go_cache}}
    cache_root="$(pwd)/{{cache_dir}}"; GOCACHE="${GOCACHE:-$cache_root/go-build}" GOOS=js GOARCH=wasm go build -trimpath -ldflags="-s -w" -o web/dist/microgpt.wasm ./cmd/microgpt-wasm
    cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" web/dist/wasm_exec.js
    cp web/index.html web/main.js web/dist/
    rm -rf web/dist/datasets
    cp -R web/datasets web/dist/datasets

# Run the CLI. Example: just run --steps 50 --samples 5
run *args:
    mkdir -p {{go_cache}}
    cache_root="$(pwd)/{{cache_dir}}"; GOCACHE="${GOCACHE:-$cache_root/go-build}" go run ./cmd/microgpt {{args}}

# Run local CI checks.
ci: check-formatted test lint check-tidy

# Clean generated artifacts.
clean:
    rm -f coverage.out coverage.html
    rm -rf bin
    rm -rf web/dist
    rm -rf {{cache_dir}}
